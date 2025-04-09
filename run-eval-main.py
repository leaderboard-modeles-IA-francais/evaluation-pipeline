import subprocess
import os

def main():
    subprocess.run("pwd", shell=True, executable="/bin/bash")
    subprocess.run("ls", shell=True, executable="/bin/bash")
    subprocess.run("./run-eval-main.sh", shell=True, check=True, executable="/bin/bash")

if __name__ == "__main__":
    main()
