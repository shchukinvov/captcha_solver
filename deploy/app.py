import gradio as gr
from app_utils import LabelEncoder, CHAR_LIST, preprocess_image
from onnxruntime import InferenceSession


model = "testmodel.onnx"
encoder = LabelEncoder(CHAR_LIST)
session = InferenceSession(model, providers=["CPUExecutionProvider"])


def onnx_inference(fpath: str,
                   inference_session: InferenceSession = session,
                   label_encoder: LabelEncoder = encoder) -> str:
    inputs = preprocess_image(fpath)
    input_feed = {"input.1": inputs}
    outputs = inference_session.run(output_names=None, input_feed=input_feed)[0]
    return label_encoder.decode_prediction(outputs)


input_image = [
    gr.components.Image(type="filepath", label="Input Image")
]

output_text = [
    gr.components.Textbox(type="text", label="Output sequence")
]

interface_image = gr.Interface(
    fn=onnx_inference,
    inputs=input_image,
    outputs=output_text,
    title="captcha solver",
    examples=["examples/ft5Yw.jpg", "examples/gDaMMk.png", "examples/geKml.jpg"],
    cache_examples=False,
)

interface_image.launch()
