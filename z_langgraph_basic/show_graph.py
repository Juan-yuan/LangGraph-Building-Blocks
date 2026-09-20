import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def show_graph_in_code(graph, file_name: str) -> None:
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open(file_name, "wb") as f:
            f.write(png_bytes)

        image = mpimg.imread(file_name)
        plt.imshow(image)
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")
