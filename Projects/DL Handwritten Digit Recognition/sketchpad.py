import gradio as gr
from predict import predict_image
# Custom CSS to make the label text bigger
def greet(image_dict):
    # Extract the "composite" key from the dictionary
    composite_image = image_dict["composite"]
    # composite_image.save("sketchpad_output.png")  # Save as PNG
    predicted = predict_image(composite_image)
    # print(predicted)
    return predicted #, composite_image  # Directly return the PIL image
css = """
.big-label {
    font-size: 24px; /* Adjust this value to make the label bigger */
    font-weight: bold; /* Optional: to make it bold */
}
"""
demo = gr.Interface(
    fn=greet,
    inputs=gr.Sketchpad(type="pil", brush=gr.Brush(default_size=20)),  # Ensure it returns a PIL image
    outputs=[gr.Label(num_top_classes=3, label="Predicted number is:", elem_classes=["big-label"])],
    css=css
)

if __name__ == "__main__":
    demo.launch()