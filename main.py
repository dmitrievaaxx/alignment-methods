# main.py
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--result')
    args = parser.parse_args()

    print("Hello from DataSphere!")
    print("Arguments:", sys.argv[1:])

    # создаём файл по пути, который передал DataSphere
    with open(args.result, "w") as f:
        f.write("Job completed successfully!\n")
        f.write("Data file: " + str(args.data) + "\n")
        f.write("Arguments: " + str(sys.argv[1:]) + "\n")

if __name__ == '__main__':
    main()