import streamlit as st


def docs():
    st.title("ChestInsight")
    st.markdown("### Smart Chest Xray Analysis and Report Generation")

    st.markdown(
        "This project aims to provide a smart solution for chest x-ray analysis and report generation. The project is divided into a series of steps including dicom image anonymization, image annotation, image enhancement, and report generation. The report generation part includes natural language processing and deep learning techniques adopted from [rgrg](https://github.com/ttanida/rgrg)."
    )

    st.markdown("### Installation")
    st.markdown("Clone the repository:")
    st.code("git clone https://github.com/rdiptan/ChestInsight.git")
    st.markdown("Install requirements:")
    st.code("pip install -r requirements.txt")

    st.markdown("### Usage")
    st.markdown("To use the project, run the following command:")
    st.code("streamlit run main.py")
    st.markdown("Note: All data files must be located in `.data/`")

    st.markdown("### Contributing")
    st.markdown("Contributions are welcome! Please feel free to submit a pull request.")

    st.markdown("### License")
    st.markdown(
        "This project is licensed under the MIT License - see the LICENSE file for details."
    )
