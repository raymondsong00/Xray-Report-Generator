import sys

from data import generate_data

def main(targets):
    if 'data' in targets: 
        generate_data()
    if 'train' in targets:
        #TODO: Add train call
        return 
    if 'test' in targets:
        #TODO: Add test call
        return 
    if 'eval' in targets:
        #TODO: Add eval target
    
if __name__ == "__main__":
    targets = sys.argv[:1]
    main(targets)