
def print_green(text):
    """Function to print green text on the console"""
    # Using ANSI Escape Codes to print green
    print(f"\033[32m{text}\033[0m")


if __name__ == "__main__":
    print_green("This function prints green text in the console")
