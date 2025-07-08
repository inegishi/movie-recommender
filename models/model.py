import csv



def create_subset(input_str, output_str):
    """
    create a subset of main dataset for testing purposes
    """

    with open(input_str, "r") as f_in, open(output_str, "w", newline="") as f_out:
        
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        for i, row in enumerate(reader):
            writer.writerow(row)
            if i >=1001:
                break


create_subset("dataset/full-dataset/ratings.csv","dataset/ratings_subset.csv")