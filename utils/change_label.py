# Specify the input and output file paths
input_file = 'identity_CelebA.txt'          # Replace with your actual input file name
output_file = 'modified_labels.txt'  # The output file name

# The target label you want to assign '1' to
target_label = 2820

# Open the input file for reading and the output file for writing
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # Remove any leading/trailing whitespace
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        
        # Split the line into image filename and label
        parts = line.split()
        if len(parts) != 2:
            print(f"Skipping invalid line: {line}")
            continue
        
        img_filename, label_str = parts
        try:
            label = int(label_str)
        except ValueError:
            print(f"Invalid label '{label_str}' on line: {line}")
            continue
        
        # Modify the label: set to 1 if it matches the target_label, else -1
        if label == target_label:
            new_label = 1
        else:
            new_label = -1
        
        # Write the modified label to the output file
        outfile.write(f"{img_filename} {new_label}\n")
