import streamlit as st
import torch
import numpy as np
from PIL import Image
import tempfile
import os
import hashlib
import io
from models.unet import UNet, UNetV3
from utils.image_processing import load_tiff_image, preprocess_image, postprocess_mask
from utils.car_detection import load_car_detector, scan_car_bboxes
from utils.visualization import create_overlay, create_results_zip, display_results

def main():
    st.set_page_config(
        page_title="Сегментация спутниковых снимков",
        layout="wide"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    AVG_CAR_AREA_M2 = 8.0
    
    st.title("Сегментация спутниковых снимков")
    
    with st.sidebar:
        st.header("Настройки")
        
        model_path = "best_model_v3.pth"
        
        if not os.path.exists(model_path):
            st.warning("Модель не найдена!")
            model = None
        else:
            try:
                state = torch.load(model_path, map_location=device)
                if isinstance(state, dict):
                    if "model" in state:
                        state = state["model"]
                    elif "state_dict" in state:
                        state = state["state_dict"]
                    elif "model_state_dict" in state:
                        state = state["model_state_dict"]
                if isinstance(state, dict):
                    cleaned = {}
                    for k, v in state.items():
                        if k.startswith("module."):
                            k = k[7:]
                        cleaned[k] = v
                    state = cleaned

                use_v3 = False
                if isinstance(state, dict):
                    for k in state.keys():
                        if ".net." in k or k in ("outc.weight", "outc.bias"):
                            use_v3 = True
                            break

                if use_v3:
                    model = UNetV3(n_channels=3, n_classes=1, base=64).to(device)
                    incompat = model.load_state_dict(state, strict=False)
                    missing = getattr(incompat, "missing_keys", [])
                    unexpected = getattr(incompat, "unexpected_keys", [])
                    if missing or unexpected:
                        st.warning(
                            "Загружено с несовпадениями: "
                            f"missing={len(missing)}, unexpected={len(unexpected)}."
                        )
                else:
                    model = UNet(n_channels=3, n_classes=1).to(device)
                    if isinstance(state, dict):
                        remapped = {}
                        for k, v in state.items():
                            if k == "outc.weight":
                                k = "outc.conv.weight"
                            if k == "outc.bias":
                                k = "outc.conv.bias"
                            remapped[k] = v
                        state = remapped
                    incompat = model.load_state_dict(state, strict=False)
                    missing = getattr(incompat, "missing_keys", [])
                    unexpected = getattr(incompat, "unexpected_keys", [])
                    if missing:
                        name_to_param = dict(model.named_parameters())
                        for name in missing:
                            if name.endswith(".bias") and name in name_to_param:
                                name_to_param[name].data.zero_()
                    if missing or unexpected:
                        st.warning(
                            "Загружено с несовпадениями: "
                            f"missing={len(missing)}, unexpected={len(unexpected)}. "
                            "Для bias параметры выставлены в ноль."
                        )
                model.eval()
            except Exception as e:
                st.error(f"Ошибка загрузки модели: {e}")
                model = None
        
        threshold = st.slider(
            "Порог уверенности",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Порог для бинаризации вероятностей"
        )
        
        alpha = st.slider(
            "Прозрачность наложения",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Прозрачность маски при наложении на оригинальное изображение"
        )

        st.markdown("---")
        st.subheader("Оценка площади (м²)")

        car_model = None
        car_score_thr = None
        target_cars = None

        car_model_path = "best_cars_detection_v1.pt"

        if not os.path.exists(car_model_path):
            st.warning("Модель детекции машин не найдена!")
        else:
            try:
                car_model = load_car_detector(car_model_path, device)
            except Exception as e:
                st.error(f"Ошибка загрузки модели машин: {e}")
                car_model = None

        car_score_thr = st.slider(
            "Порог детекции машин",
            min_value=0.1,
            max_value=0.95,
            value=0.8,
            step=0.05
        )

        target_cars = st.radio(
            "Сколько машин искать для усреднения",
            options=[1, 10, 25],
            index=0,
            horizontal=True
        )
        if target_cars in (10, 25):
            st.warning("Поиск 10 и 25 машин дольше по времени, но точнее для оценки площади. Подождите немного")

    if model is None:
        st.error("Модель не загружена!")
        return
    
    uploaded_file = st.file_uploader(
        "Загрузите TIFF файл",
        type=['tif', 'tiff'],
        help="Выберите файл спутникового снимка в формате TIFF"
    )
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_id = hashlib.md5(file_bytes).hexdigest()

        if "image_cache" not in st.session_state:
            st.session_state["image_cache"] = {}
        if "seg_cache" not in st.session_state:
            st.session_state["seg_cache"] = {}
        if "car_cache" not in st.session_state:
            st.session_state["car_cache"] = {}

        image_cache = st.session_state["image_cache"]
        seg_cache = st.session_state["seg_cache"]
        car_cache = st.session_state["car_cache"]

        if file_id not in image_cache:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name
            try:
                with st.spinner("Загрузка изображения..."):
                    original_image, original_shape, processed_shape, full_image_u8 = load_tiff_image(
                        tmp_path,
                        return_full=True
                    )
                image_cache[file_id] = {
                    "original_image": original_image,
                    "original_shape": original_shape,
                    "processed_shape": processed_shape,
                    "full_image_u8": full_image_u8,
                }
            except Exception as e:
                st.error(f"Ошибка при обработке файла: {str(e)}")
                return
            finally:
                os.unlink(tmp_path)

        image_data = image_cache[file_id]
        original_image = image_data["original_image"]
        original_shape = image_data["original_shape"]
        processed_shape = image_data["processed_shape"]
        full_image_u8 = image_data["full_image_u8"]

        seg_key = (file_id, float(threshold))
        if seg_key not in seg_cache:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Подготовка изображения...")
            progress_bar.progress(20)

            image_tensor = preprocess_image(original_image).to(device)

            status_text.text("Выполнение сегментации...")
            progress_bar.progress(50)

            with torch.no_grad():
                output = model(image_tensor)
                prediction = torch.sigmoid(output)

            status_text.text("Постобработка результатов...")
            progress_bar.progress(80)

            mask_pred = prediction.squeeze().cpu().numpy()
            final_mask = postprocess_mask(mask_pred, original_shape, processed_shape, threshold)

            progress_bar.progress(100)
            status_text.text("Сегментация завершена!")
            progress_bar.empty()
            status_text.empty()

            building_pixels = int(np.sum(final_mask > 0))
            total_pixels = int(final_mask.size)
            building_percentage = (building_pixels / max(total_pixels, 1)) * 100

            seg_cache[seg_key] = {
                "final_mask": final_mask,
                "building_pixels": building_pixels,
                "total_pixels": total_pixels,
                "building_percentage": building_percentage,
            }

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Загруженное изображение")
            if original_image is not None:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(original_image)
                ax.axis('off')
                st.pyplot(fig)

                st.info(f"""
                **Размер:** {original_shape[1]} × {original_shape[0]} пикселей  
                **Обработанный размер:** {processed_shape[1]} × {processed_shape[0]} пикселей  
                """)

        with col2:
            st.subheader("Результаты сегментации")
            result_plot = st.empty()
            stats_box = st.empty()
            area_box = st.empty()
            warn_box = st.empty()

        if seg_key not in seg_cache:
            with col2:
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Подготовка изображения...")
                progress_bar.progress(20)

                image_tensor = preprocess_image(original_image).to(device)

                status_text.text("Выполнение сегментации...")
                progress_bar.progress(50)

                with torch.no_grad():
                    output = model(image_tensor)
                    prediction = torch.sigmoid(output)

                status_text.text("Постобработка результатов...")
                progress_bar.progress(80)

                mask_pred = prediction.squeeze().cpu().numpy()
                final_mask = postprocess_mask(mask_pred, original_shape, processed_shape, threshold)

                progress_bar.progress(100)
                status_text.text("Сегментация завершена!")
                progress_bar.empty()
                status_text.empty()

                building_pixels = int(np.sum(final_mask > 0))
                total_pixels = int(final_mask.size)
                building_percentage = (building_pixels / max(total_pixels, 1)) * 100

                seg_cache[seg_key] = {
                    "final_mask": final_mask,
                    "building_pixels": building_pixels,
                    "total_pixels": total_pixels,
                    "building_percentage": building_percentage,
                }

        seg_result = seg_cache[seg_key]
        final_mask = seg_result["final_mask"]
        building_pixels = seg_result["building_pixels"]
        total_pixels = seg_result["total_pixels"]
        building_percentage = seg_result["building_percentage"]

        overlay = create_overlay((original_image * 255).astype(np.uint8), final_mask, alpha)
        fig = display_results(original_image, final_mask, overlay, threshold, alpha)
        result_plot.pyplot(fig)

        building_area_m2 = None
        cars_detected = 0
        avg_bbox_area_px = None

        stats_text = f"""
        **Результаты сегментации:**  
        • Всего пикселей: {total_pixels:,}  
        • Пикселей застройки: {building_pixels:,}  
        • Покрытие зданиями: {building_percentage:.2f}%  
        """
        if building_area_m2 is not None:
            stats_text += f"• Площадь застройки: {building_area_m2:,.2f} м²\n"
        stats_box.success(stats_text)

        if car_model is None:
            warn_box.warning("Оценка площади недоступна: модель детекции машин не загружена.")
        else:
            car_key = (file_id, float(car_score_thr), int(target_cars))
            if car_key not in car_cache:
                with col2:
                    with st.spinner("Детекция машин для оценки площади..."):
                        car_progress = st.progress(0)
                        car_status = st.empty()

                        def _car_progress(done, total, found):
                            if total > 0:
                                car_progress.progress(min(done / total, 1.0))
                            car_status.text(f"Тайлов обработано: {done}/{total} | Машин найдено: {found}/{target_cars}")

                        all_boxes, tiles_done, _ = scan_car_bboxes(
                            full_image_u8,
                            car_model,
                            device,
                            score_thr=float(car_score_thr),
                            target_cars=int(target_cars) if target_cars is not None else None,
                            progress_cb=_car_progress
                        )

                        car_progress.empty()
                        car_status.empty()

                        cars_detected = int(len(all_boxes))
                        if cars_detected > 0:
                            bbox_areas_px = (all_boxes[:, 2] - all_boxes[:, 0]) * (all_boxes[:, 3] - all_boxes[:, 1])
                            avg_bbox_area_px = float(np.mean(bbox_areas_px))

                        car_cache[car_key] = {
                            "cars_detected": cars_detected,
                            "avg_bbox_area_px": avg_bbox_area_px,
                        }

            car_result = car_cache[car_key]
            cars_detected = car_result["cars_detected"]
            avg_bbox_area_px = car_result["avg_bbox_area_px"]

            if cars_detected > 0 and avg_bbox_area_px is not None:
                m2_per_pixel = float(AVG_CAR_AREA_M2) / max(avg_bbox_area_px, 1e-6)
                building_area_m2 = float(building_pixels) * m2_per_pixel

            if building_area_m2 is not None:
                area_box.info(f"""
                **Оценка площади (м²):**  
                • Машин обнаружено: {cars_detected}  
                • Средняя площадь bbox (px²): {avg_bbox_area_px:,.2f}  
                • Оцененная площадь застройки: {building_area_m2:,.2f} м²
                """)
                stats_text = f"""
                **Результаты сегментации:**  
                • Всего пикселей: {total_pixels:,}  
                • Пикселей застройки: {building_pixels:,}  
                • Покрытие зданиями: {building_percentage:.2f}%  
                • Площадь застройки: {building_area_m2:,.2f} м²
                """
                stats_box.success(stats_text)
                if target_cars is not None and cars_detected < int(target_cars):
                    warn_box.warning(f"Найдено только {cars_detected} из {target_cars} машин. Прошлись по всем тайлам.")
            else:
                warn_box.warning("Не удалось оценить площадь: машины не найдены.")

        st.markdown("---")
        st.subheader("Сохранение результатов")

        col_save1, col_save2, col_save3 = st.columns(3)

        with col_save1:
            mask_pil = Image.fromarray((final_mask * 255).astype(np.uint8))
            mask_buf = io.BytesIO()
            mask_pil.save(mask_buf, format="PNG")
            mask_bytes = mask_buf.getvalue()

            st.download_button(
                label="Скачать маску (PNG)",
                data=mask_bytes,
                file_name="segmentation_mask.png",
                mime="image/png",
                help="Скачать бинарную маску сегментации"
            )

        with col_save2:
            overlay_pil = Image.fromarray(overlay.astype(np.uint8))
            overlay_buf = io.BytesIO()
            overlay_pil.save(overlay_buf, format="PNG")
            overlay_bytes = overlay_buf.getvalue()

            st.download_button(
                label="Скачать наложение (PNG)",
                data=overlay_bytes,
                file_name="segmentation_overlay.png",
                mime="image/png",
                help="Скачать изображение с наложенной маской"
            )

        with col_save3:
            zip_data = create_results_zip(original_image, final_mask, overlay)

            st.download_button(
                label="Скачать все (ZIP)",
                data=zip_data,
                file_name="segmentation_results.zip",
                mime="application/zip",
                help="Скачать все результаты в ZIP архиве"
            )
    
   

if __name__ == "__main__":
    main()
