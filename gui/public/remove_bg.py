from PIL import Image# type: ignore
import os

def remove_background(input_path, output_path, tolerance=50):
    try:
        img = Image.open(input_path).convert("RGBA")
        datas = img.getdata()
        
        # Get the background color from the top-left pixel
        # The generated image background is usually solid.
        bg_color = datas[0] 

        newData = []
        for item in datas:
            # Check if pixel is close to background color
            if (abs(item[0] - bg_color[0]) < tolerance and 
                abs(item[1] - bg_color[1]) < tolerance and 
                abs(item[2] - bg_color[2]) < tolerance):
                # Replace with transparent pixel
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)

        img.putdata(newData)
        img.save(output_path, "PNG")
        print(f"Successfully processed {input_path}")
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    input_file = r"c:\Users\aayus\Desktop\sign-language-translator\gui\public\signmate_logo.png"
    output_file = r"c:\Users\aayus\Desktop\sign-language-translator\gui\public\signmate_logo.png"
    
    if os.path.exists(input_file):
        # We'll write to a temp file first then replace to be safe
        temp_file = r"c:\Users\aayus\Desktop\sign-language-translator\gui\public\signmate_logo_temp.png"
        remove_background(input_file, temp_file, tolerance=30)
        
        if os.path.exists(temp_file):
            os.replace(temp_file, output_file)
            print("Done replacing.")
    else:
        print("Input file not found.")
