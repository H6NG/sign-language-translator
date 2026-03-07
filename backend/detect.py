import platform

def get_os():
    system = platform.system()

    if system == "Darwin":
        return "mac"
    elif system == "Windows":
        return "windows"
    else:
        return "linux"

print("Running on:", get_os())
