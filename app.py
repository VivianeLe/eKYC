import time
import cv2
import streamlit as st
import SessionState
import numpy as np
from PIL import Image
from pipeline import InferencePipeline
state = SessionState.get(result_text="", res="", prob_positive=0.0, prob_negative= 0.0, initial=True, img_drawed=None, img_cropped=None, reg_text_time=None)


def main():
    model = load_model()
    st.title("Demo nhận dạng văn bản tiếng Việt")
    # Load model

    pages = {
        'CMND': page_cmnd

    }

    st.sidebar.title("Application")
    page = st.sidebar.radio("Chọn ứng dụng demo:", tuple(pages.keys()))

    pages[page](state, model)

    # state.sync()


@st.cache(allow_output_mutation=True)  # hash_func
def load_model():
    print("Loading model ...")
    model = InferencePipeline(device='cpu')
    return model


def page_cmnd(state, model):
    st.header("Nhận dạng văn bản từ CMND")

    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
        # Convert the uploaded image to OpenCV format
        pil_image = Image.open(img_file_buffer)
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # CMND detection
        t1 = time.time()
        
        # Run inference model
        result_text, img_drawed_box = model.run(cv_image)

        if img_drawed_box is not None and img_drawed_box.size != 0:
            # Convert the output image back to RGB if valid
            img_drawed_box = cv2.cvtColor(img_drawed_box, cv2.COLOR_BGR2RGB)
            state.img_drawed = img_drawed_box
        else:
            st.error("The output image is empty. Please check the model inference step.")
            return

        state.result_text = result_text
        state.reg_text_time = time.time() - t1

        # Display results
        col1, col2 = st.columns(2)  # Replace beta_columns with columns
        with col2:
            if state.result_text:
                st.json(state.result_text)
                st.success("Time: %.2f seconds" % (state.reg_text_time))
            else:
                st.error("Not detected CMND")
                
        with col1:
            if state.img_drawed is not None:
                st.image(state.img_drawed, use_column_width=True)


        # if state.img_cropped is not None:
        #     st.title("Chi tiết:")
        #     for idx, img in enumerate(state.img_cropped):
        #         st.image(img, caption=state.result_text[idx])
        #         st.empty()

if __name__ == "__main__":
    main()