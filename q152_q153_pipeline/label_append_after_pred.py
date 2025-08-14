### Re-Label Based on Accurate Predictions From Scored Subgroup ###
# (Ran after the first prediction and never used again. Provided 62 additional sentences for labeling. Potential reduced number of matched sentences could be due to hash mismatches. 
# Did not use the full unga_wvs7_hashed_corpus.csv, but rather the top scored sentences from the WVC corpus.)

import pandas as pd

def append_accurate_predictions(
    selected_hashes,
    predictions_path="predictions/q152_predictions.csv",
    top_scored_path="top_scored_sentences.csv",
    labeled_path="mmr_iqr_labeled/Q152_mmr_selected_labeled.csv"
):
    # Load all predicted hashes from the predictions file
    pred_df = pd.read_csv(predictions_path)
    all_pred_hashes = set(pred_df['embedding_hash'].unique())
    print(f"Total hashes in predictions: {len(all_pred_hashes)}")
    
    # Filter selected hashes to only those present in the predictions file
    valid_hashes = [h for h in selected_hashes if h in all_pred_hashes]
    print(f"Valid selected hashes found in predictions: {len(valid_hashes)}")
    
    if not valid_hashes:
        print("No valid hashes found in predictions to process.")
        return
    
    # Find matching rows in top scored sentences
    top_df = pd.read_csv(top_scored_path)
    matched_rows = top_df[top_df['embedding_hash'].isin(valid_hashes)].copy()
    
    if matched_rows.empty:
        print("No matching rows found in top_scored_sentences.csv for the valid hashes.")
        return
    
    matched_rows['match'] = 1
    
    # Load labeled data
    labeled_df = pd.read_csv(labeled_path)
    if 'match' not in labeled_df.columns:
        labeled_df['match'] = 0
    
    # Append and save
    combined_df = pd.concat([labeled_df, matched_rows], ignore_index=True)
    combined_df.to_csv(labeled_path, index=False)
    print(f"Appended {len(matched_rows)} rows to labeled data and saved to '{labeled_path}'.")

if __name__ == "__main__":
    # Look for specific hashes to append
    selected_hashes = [
        "66291e59a4ca8fc0794edd66468775e13d22d0004ba057ec2d2c791abe1ad484",
        "40098797a6b0b69a993241c6030c445580b023206fa4f9d1d9fe4e2161322ce8",
        "af2bd3882190394d4a2537fb949c2af16dba34a5125a16b365128d01524110bd",
        "14b223cdb0c3f0281d5f87794c3a3b0db559b922d814d2130460712c91e7450c",
        "3a284e2f40577adcf1c04f3bbff920593d77878c2ec1667dc6da71e01a10a369",
        "23ea7b3e49ca1225d28bb13d037b56d15f12d9d6a13b2885d0b876a7e5247979",
        "318e2422c96f57af87018fcff14cdcd35dde9a01fe3c3af465076555a8bb2f7f",
        "ba0ef448ce5293eade307c68fc9b8004848cfe7f7ce7af0125b0a3ee0737907e",
        "981c2ea2d8a894405c65b3f4528fcddfa086fcea32715c2256086aac32b20f8b",
        "24c7004f5a6de53e40ee9ad1f59880ddfb5304d353110377f6483d1e8b314f3b",
        "998890dee86fff9fecffe64cfbf795898b59371c44694a9b5e0e8f1402b8b190",
        "43a8d628098977583ac4d40403f535d66546aeca4d8912ec6661cdeedfd122c5",
        "7dc45a0c7b8378749e478a5cfa7dc4d6f6a0c3af1c84bcc0b67c9d2de09f2439",
        "8a027eb2e9069ac2e76cff07fa5c33f257b55eb6a9b0c5d7025de303093c054e",
        "2773daa691fdfee8877d0bd153f254ef4c8b722e3d4843d579f25288b45cd6dc",
        "cba1dd4f96e6db2222f937120548af50ad52bc700dfdd181fe534728099238f3",
        "636ab65f8167c84f63c946c41a5e9b2c4fc23b83de6ae6411c078b97cc001640",
        "b8d6c2b37a76ca2a7b7e4aad46e101a8f5aa45609961053aa3e82842864553ff",
        "23779b50fb01082244022a2d802ad552d772afdb6c97214b20ef0a2321d76c54",
        "650c129c964c70572b5a2ce0ad46e8e3dc69109528ecf4a7de48c1061b70a14d",
        "49ad5aa61e4eb1d822dff4f6bf2b4419ec60981ec2dd124485d80e0da7dc62cf",
        "2683eee7ba33f526bba8dc324b8ee9ab15562efbd6227d8a4d77ac2e82c538b7",
        "64f2a77b0abf5a125f3dc71fa4c716f3bd7847feb3ed9663ef70c5fb90b8fa11",
        "9540bb047c4297324d647b70e0cc02289219e55d2af32e819135e97d9b187a28",
        "bad7cb75f42ef084a7e3c18cf683dd13ab62ddc7b9c5e28f0b5eaf20cd9a528e",
        "1894714892dde370c1c6fb0b87b9a5703c51e232d6aef0452560386b49fca8db",
        "2fdf20b04378bd5e5af20e3c67074d625178b49f17648412f4d51a3d6df6dc8b",
        "c97eb96da3b63a6eb8c8c3624c8fee700961d64fd6d67fd3baeb4967bcbbbe3a",
        "f32fe5bb3ed425de3006cdb70ac61ee5d3b7c16a37e7c2d4b844907e0ac58a30",
        "37b25d45030f9ba7b80bb5586008d8cd1b3427e9939ee9a23f3e758cd03de9d8",
        "b7ed0f3c2588df989369d054177e430e60dbc40bbcdf2580b3befb6d76d6411d",
        "35804323d3c131b26ec665eb85cb5dd6d686160eecc93f38c7400a54d5298bcd",
        "ee1887d935c428fa0027116a565562d0780245f2b770d8962e428d294db5c8d4",
        "313516b68b3bec797133702c42957a10072c480d0b9a9eea0210029e61cafa5a",
        "bc120f3b323ac9fab13f93a34d036002c46f4987bffe5ba6a99700e8fff31664",
        "68e271ad5c6eaa11023e700628f62d482eb0304433607e2bc2d19ebf04e6e724",
        "07defb5fac428a64e7ef3f26d744bd73258668a0b41cdee3f6497ec96a933ea9",
        "4246ee2bc09eed82bc3df5c385dcddcccc949096ef7a6b99333c0b864d00de02",
        "0696f1b1b37f41e432634434f5c8c04f897d86c6d41e5165f750dc406bacc9ae",
        "aad6d508bbc28f1233326d2589645e339217a436dddbeb7f226cd90f50c37b7a",
        "73511b13c492064685a8116f2d575ede8fa2c6406ea382a2be2a8519d550b869",
        "5fd97a6ba0dd6fac4aa13735c5b4a63ca8b79a1ad58fbf4fa15491555a11c69e",
        "f87812f95d23572a52b565f6bcf88f81c6646f01a8fd54aa96e46d1d52dc8b44",
        "3c12084fc15f54c452e12e5901772ae36e932938e95d56b80ba83deed90afe8e",
        "4d320208130213ed4d39e4c112db97d7faab205cf4992dffbbd03f5494eb6219",
        "c9ac3afdbedff09bdf829958de17b1754c8fa40aa6ba1d59c0872d7733ef309a",
        "04965a2c9eae1289f0261e4a61788301f9e41fc00d39fd8c1c56286f55d5c409",
        "efc368479aa8530533a425ae9c64f24ee486e70fe7007498aac0f3e460f0b483",
        "d7cddc94c445edf4be2dc7e86db411ebc0a8f547459befad36e76b821b229668",
        "125be74f225d0285e79cc72449297b825fd8d269b89f25a0048d655db8a368f5"
    ]       
    append_accurate_predictions(selected_hashes)
