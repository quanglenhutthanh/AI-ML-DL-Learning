import gradio as gr
from predict import predict_image

def greet(image_dict):
    # Extract the "composite" key from the dictionary
    composite_image = image_dict["composite"]
    # composite_image.save("sketchpad_output.png")  # Save as PNG
    predicted = predict_image(composite_image)
    # print(predicted)
    return composite_image, predicted  # Directly return the PIL image

demo = gr.Interface(
    fn=greet,
    inputs=gr.Sketchpad(type="pil"),  # Ensure it returns a PIL image
    outputs=["image",gr.Label(num_top_classes=3)]
)

if __name__ == "__main__":
    demo.launch()