import os
import argparse
from openbabel import openbabel as ob

def extract_and_save_molecules_as_cif(file_path, output_directory):
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Initialize the converter
    obConversion = ob.OBConversion()
    obConversion.SetInFormat("cif")
    obConversion.SetOutFormat("cif")

    # Create an OBMol object to store the whole structure
    mol = ob.OBMol()

    # Read the file
    print(f"Processing file: {file_path}")
    if not obConversion.ReadFile(mol, file_path):
        print("Failed to read the file")
        return
    
    # Get the original unit cell parameters
    unit_cell = mol.GetData(ob.UnitCell)
    unit_cell = ob.toUnitCell(unit_cell)

    # Get the molecular components
    components = mol.Separate()
    extracted_molecules = []

    # Iterate through all separated molecules
    for i, comp in enumerate(components, 1):
        comp_mol = ob.OBMol(comp)
        new_unit_cell = ob.OBUnitCell()
        new_unit_cell.SetData(unit_cell.GetA(), unit_cell.GetB(), unit_cell.GetC(),
                            unit_cell.GetAlpha(), unit_cell.GetBeta(), unit_cell.GetGamma())
        comp_mol.CloneData(new_unit_cell)
        cif_filename = os.path.join(output_directory, f"molecule_{i}.cif")
        # Use the converter to save the molecule as a CIF file
        obConversion.WriteFile(comp_mol, cif_filename)

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Split CIF files and save molecules as individual CIF files")
    parser.add_argument('cif_file_path', type=str, help="Path to the input CIF folder")
    parser.add_argument('output_directory', type=str, help="Path to the output directory")
    
    args = parser.parse_args()

    cif_file_path = args.cif_file_path
    output_directory = args.output_directory

    # Get all .cif files in the input folder
    cif_lists = os.listdir(cif_file_path)

    # Iterate through all CIF files in the specified folder and split them
    for cif_file in cif_lists:
        if cif_file.endswith(".cif"):
            print(f"Processing {cif_file}")
            input_path = os.path.join(cif_file_path, cif_file)
            output_path = os.path.join(output_directory, cif_file.split('.')[0])
            extract_and_save_molecules_as_cif(input_path, output_path)

if __name__ == "__main__":
    main()


# cif_file_path = '/opt/wty/hof_synthesis/HOF/data_other/hof_database_cifs'
# output_directory = '/opt/wty/hof_synthesis/HOF/MOFDiff/mofdiff/data/hof_data'
# cif_database = '/opt/wty/hof_synthesis/HOF/data_other/hof_database_cifs'