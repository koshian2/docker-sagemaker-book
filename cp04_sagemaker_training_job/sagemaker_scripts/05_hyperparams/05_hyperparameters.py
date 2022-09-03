import os
import json

def main():
    environ_str = json.dumps(dict(os.environ), 
        indent=4, sort_keys=True, separators=(',', ': '))
    print(environ_str)

if __name__ == "__main__":
    main()
