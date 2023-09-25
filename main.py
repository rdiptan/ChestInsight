import json
import textwrap
import warnings
import streamlit as st
st.set_page_config(layout="wide")
from streamlit_option_menu import option_menu   
from streamlit_extras.colored_header import colored_header
from image_annotation import run_cls, dataframe_annotation
from dicom_viewer_and_annon import anonymize_dicom_file, dicom_viewer
warnings.filterwarnings("ignore")

def features():
    #  headings and tabs creation
    DicomAnonymizationTab, ImgAnnotationTab, ImgEnhancementTab, FinetuningTab, PredictionTab,  = st.tabs(["Dicom Image Explorer and Anonymizer", "Annonate Images and Generate Datasets",
                                        "SOTA Image Enhancement", " FineTune DL models", "Prediction and Report Generation with Grad-Cam Explainer", ])
    
    # DicomAnnonymizationTab: Here is the entry point for dicom image explorer
    with DicomAnonymizationTab:
        DcmAnonTab, DcmViewTab, = st.tabs(["Dicom Image Anonymization", "Dicom Viewer",])
        # DcmAnonTab: Here is the entry point for dicom image annonymization
        with DcmAnonTab:
            dicom_file = st.file_uploader("Upload a DICOM file", key="dicom_annon")
            if dicom_file is not None: 
                # Anonymize DICOM file
                anonymized_dicom_dataset = anonymize_dicom_file(dicom_file)
                desired_file_name = st.text_input("Enter the desired file name for the anonymized DICOM file, press enter to save: ")
                if st.button("Save anonymized dicom file"):
                    # Add the .dcm extension if not provided
                    if desired_file_name is not None:
                        if not desired_file_name.endswith('.dcm'):
                            desired_file_name += '.dcm'
                            # Save the anonymized DICOM file
                            anonymized_dicom_dataset.save_as(f'data/{desired_file_name}')
                            st.success('Anonymized Dicom file Saved!')
            # DcmViewTab: Here is the entry point for dicom image viewer
        with DcmViewTab:
            dicom_file = st.file_uploader("Upload a DICOM file", key="dicom_viewer")
            if dicom_file is not None:
                dicom_viewer(dicom_file)
    
    # ImgenhancementTab: Here is the entry point for SOTA Image enhancement
    with ImgEnhancementTab:
        st.write('wait a minute')
    
    # ImgAnnotationTab: Here is the entry point for image annotation
    with ImgAnnotationTab:
        custom_labels = ["", "lesion", "positive", "negative", "tumor", None]
        path = st.text_input('Enter the path to image folder', key="clsTab_path")
        if path:
            select_label, report = run_cls(f"{path}", custom_labels)
            dataframe_annotation(f'{path}/*.jpg', custom_labels, select_label, report)
    
    # FinetuningTab: Here is the entry point for model finetuning
    with FinetuningTab:
        st.write('wait a minute')
    
    # PredictionTab: Here is the entry point for report generation
    with PredictionTab:
        st.write('wait a minute')


def main():
    #st.markdown("<h2 style='text-align: center; color: blue;'>Smart Chest-Xray Analysis and Report Generation!</h2>", unsafe_allow_html=True)

    with st.sidebar:
        choice = option_menu("Main Menu", ["About", "Try out!"], 
            icons=['house', 'fire'], menu_icon="cast", default_index=0,
        styles={
        "container": {"padding": "0!important", "background-color": "#262730"},
        "icon": {"color": "white", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#3739b5"},
        "nav-link-selected": {"background-color": "#1f8ff6"},})  

    if choice == "About":
        st.image('utils/CI 1.png', use_column_width=True)   
        st.write('Write short documentation here')

    elif choice == "Try out!":
        colored_header(
        label="CHEST-INSIGHT: Smart Chest-Xray Analysis and Report Generation! ",
        description="Use the tabs below to tryout our dedicated tools",
        color_name="violet-70",)
        features()
        

if __name__ == "__main__":
    main()
