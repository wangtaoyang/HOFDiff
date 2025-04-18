import random
import sys

def split_data(m, train_file='/data/user2/wty/HOF/MOFDiff/splits/train_split.txt', val_file='/data/user2/wty/HOF/MOFDiff/splits/val_split.txt'):
    # Generate a list of numbers from 0 to m-1
    indices = list(range(m))
    
    # Shuffle the list to ensure randomness
    random.shuffle(indices)
    
    # Calculate the split index
    split_index = int(m * 0.8)  # 80% for training, 20% for validation
    
    # Split the indices
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]
    
    # Write to train_split.txt
    with open(train_file, 'w') as train_f:
        for index in train_indices:
            train_f.write(f"{index}\n")
    
    # Write to val_split.txt
    with open(val_file, 'w') as val_f:
        for index in val_indices:
            val_f.write(f"{index}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <number_of_samples>")
        sys.exit(1)
    
    m = int(sys.argv[1])
    print(m)
    split_data(m)
