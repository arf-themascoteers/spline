import constants
import pandas as pd
import os
import spec_utils

BASE_DIR = constants.LUCAS_PATH
out_file = constants.DATASET

topsoil_df = pd.read_csv(constants.TOPSOIL_FILE)
out = open(out_file, "w")
out.write("id,oc,phh,phc,ec,caco3,p,n,k,elevation,stones,lc1,lu1")
out.write(spec_utils.get_wavelengths_str())
out.write("\n")
done = {}

for path in constants.SPECTRA_FILES:
    a_df = pd.read_csv(path)
    irrelevant_cols = ["source", "SampleID", "NUTS_0", "SampleN"]
    spectra_rows = a_df.drop(columns=irrelevant_cols, axis=1)
    df_group_object = spectra_rows.groupby(['PointID'])
    df_mean = df_group_object.mean().reset_index()
    filename = os.path.basename(path)
    for i in range(len(df_mean)):
        spectra_row = df_mean.iloc[i]
        point_id = spectra_row['PointID']
        rows = (topsoil_df.loc[topsoil_df['Point_ID'] == point_id])

        if len(rows) == 0:
            print(f"Not found in topsoil {filename},{point_id}")
            continue

        if len(rows) > 1:
            print(f"Multiple found in topsoil {filename},{point_id}")

        topsoil_row = rows.iloc[0]
        lc1_desc = topsoil_row['LC1_Desc'].replace(",","-")
        lu1_desc = topsoil_row['LU1_Desc'].replace(",","-")
        out.write(f"{topsoil_row['Point_ID']},{topsoil_row['OC']},{topsoil_row['pH(H2O)']},"
                  f"{topsoil_row['pH(CaCl2)']},{topsoil_row['EC']},"
                  f"{topsoil_row['CaCO3']},{topsoil_row['P']},{topsoil_row['N']},{topsoil_row['K']},"
                  f"{topsoil_row['Elevation']},"
                  f"{topsoil_row['Soil_Stones']},{lc1_desc},{lu1_desc}")

        for wavelength in spec_utils.wavelengths_itr():
            out.write(f",{round(spectra_row[wavelength], 4)}")
        out.write("\n")
        done[point_id] = filename
        if len(done)%100 == 0:
            print(f"Done {len(done)}")


out.close()
print("done")

