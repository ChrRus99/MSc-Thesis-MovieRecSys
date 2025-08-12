import os
import nest_asyncio
import logging
import io
from IPython.display import Image, display
from PIL import Image as PILImage
from langchain_core.runnables.graph import MermaidDrawMethod # Import the enum for draw methods
from langgraph.graph.state import CompiledStateGraph

# Remove Pyppeteer specific revision setting
# PYPPETEER_CHROMIUM_REVISION = '1263111'
# os.environ['PYPPETEER_CHROMIUM_REVISION'] = PYPPETEER_CHROMIUM_REVISION


def plot_langgraph_graph(graph: CompiledStateGraph, scale_factor: float=0.35):
    """
    Generates and displays a visualization of a LangGraph graph.

    Attempts to use the default Mermaid rendering method first. If that fails, it falls back to 
    using playwright and resizes the image.

    Args:
        graph: The LangGraph graph object to visualize.
        scale_factor: The factor by which to scale the image if playwright is used. Defaults to 0.35.
    """
    # Apply nest_asyncio patch if running in an environment like Jupyter where it's needed
    nest_asyncio.apply()

    print("Attempting to generate and display the graph visualization...")

    # Logging Suppression
    original_disable_level = logging.root.manager.disable # Store current disable level
    logging.disable(logging.CRITICAL) # Disable all logs below CRITICAL globally

    try:
        try:
            # Try the default rendering method first
            img_bytes_default = graph.get_graph().draw_mermaid_png()
            display(Image(img_bytes_default))
            print("Graph visualization successfully displayed using the default method.")
            return # Exit function after successful display

        except Exception as e:
            print(f"Default method failed: {e}. Trying playwright fallback...")

        try:
            # Generate the graph visualization using playwright
            img_bytes = graph.get_graph().draw_mermaid_png(
                draw_method=MermaidDrawMethod.PLAYWRIGHT # Use PLAYWRIGHT instead of PYPPETEER
            )

            try:
                # Load image bytes into Pillow to resize
                img = PILImage.open(io.BytesIO(img_bytes))

                # Calculate new dimensions
                original_width, original_height = img.size
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)

                # Resize the image using high-quality downsampling
                resized_img = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)

                # Save resized image back to bytes in PNG format
                buffer = io.BytesIO()
                resized_img.save(buffer, format="PNG")
                resized_img_bytes = buffer.getvalue()

                display(Image(resized_img_bytes))
                print(f"Graph visualization successfully displayed using playwright (resized to {int(scale_factor*100)}%).")

            except ImportError:
                print("Pillow library not found. Please install it (`pip install Pillow`) to resize the image.")
                # Display the original image if resizing fails due to missing library
                display(Image(img_bytes))
                print("Displayed original, unresized image instead (Pillow not found).")
            except Exception as resize_err:
                 print(f"Resizing failed: {resize_err}")
                 # Display the original image if resizing fails
                 display(Image(img_bytes))
                 print("Displayed original, unresized image instead (resizing error).")

        except ImportError as import_err:
             print(f"Playwright import failed: {import_err}") # Update error message
             print("Please ensure playwright is installed (`pip install playwright`) and browsers are installed (`playwright install`).") # Update instructions
        except Exception as e_playwright: # Update variable name
            print(f"Playwright fallback failed: {e_playwright}") # Update error message
            # Remove chromium revision message
            print("Ensure playwright is installed and browsers are set up (`playwright install`).")
            print("Graph visualization could not be generated.")

    finally:
        # Restore Logging
        #logging.disable(original_disable_level) # Restore the original disable level
        print("Logging settings restored.")
