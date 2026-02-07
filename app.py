import streamlit as st
import torch
import numpy as np
from PIL import Image
import tempfile
import os
from models.unet import UNet
from utils.image_processing import load_tiff_image, preprocess_image, postprocess_mask
from utils.visualization import create_overlay, create_results_zip, display_results

def main():
    st.set_page_config(
        page_title="Сегментация спутниковых снимков",
        layout="wide"
    )
    
    st.title("Сегментация спутниковых снимков")
    
    with st.sidebar:
        st.header("Настройки")
        
        model_path = "unet.pth"
        
        if not os.path.exists(model_path):
            st.warning("Модель не найдена!")
            model = None
        else:
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = UNet(n_channels=3, n_classes=1).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
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
        
        
    
    if model is None:
        st.error("Модель не загружена!")
        return
    
    uploaded_file = st.file_uploader(
        "Загрузите TIFF файл",
        type=['tif', 'tiff'],
        help="Выберите файл спутникового снимка в формате TIFF"
    )
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Загруженное изображение")
                
                with st.spinner("Загрузка изображения..."):
                    original_image, original_shape, processed_shape = load_tiff_image(tmp_path)
                    
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
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Подготовка изображения...")
                progress_bar.progress(20)
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
                
                overlay = create_overlay((original_image * 255).astype(np.uint8), final_mask, alpha)
                
                status_text.text("Визуализация результатов...")
                progress_bar.progress(90)
                
                fig = display_results(original_image, final_mask, overlay, threshold, alpha)
                st.pyplot(fig)
                
                progress_bar.progress(100)
                status_text.text("Обработка завершена!")
                
                building_pixels = np.sum(final_mask > 0)
                total_pixels = final_mask.size
                building_percentage = (building_pixels / total_pixels) * 100
                
                st.success(f"""
                **Результаты сегментации:**  
                • Всего пикселей: {total_pixels:,}  
                • Пикселей застройки: {building_pixels:,}  
                • Покрытие зданиями: {building_percentage:.2f}%
                """)
            
            st.markdown("---")
            st.subheader("Сохранение результатов")
            
            col_save1, col_save2, col_save3 = st.columns(3)
            
            with col_save1:
                mask_pil = Image.fromarray(final_mask * 255)
                mask_bytes = mask_pil.tobytes()
                
                st.download_button(
                    label="Скачать маску (PNG)",
                    data=mask_bytes,
                    file_name="segmentation_mask.png",
                    mime="image/png",
                    help="Скачать бинарную маску сегментации"
                )
            
            with col_save2:
                overlay_pil = Image.fromarray(overlay)
                overlay_bytes = overlay_pil.tobytes()
                
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
            
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {str(e)}")
        
        finally:
            os.unlink(tmp_path)
    
   

if __name__ == "__main__":
    main()