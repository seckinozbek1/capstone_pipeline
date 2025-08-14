### Re-Label Based on Accurate Predictions From Scored Subgroup ###
# (Ran after the first prediction and never used again. Provided 62 additional sentences for labeling. Potential reduced number of matched sentences could be due to hash mismatches. 
# Did not use the full unga_wvs7_hashed_corpus.csv, but rather the top scored sentences from the WVC corpus.)

import pandas as pd

def append_accurate_predictions(
    selected_hashes,
    predictions_path="predictions/q154_predictions_top_score.csv",
    top_scored_path="../top_scored_sentences.csv",
    labeled_path="Q154_mmr_selected_labeled.csv"
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
    combined_df = combined_df.drop_duplicates(subset='embedding_hash', keep='first')
    combined_df.to_csv(labeled_path, index=False)
    print(f"Appended {len(matched_rows)} rows to labeled data and saved to '{labeled_path}'.")

if __name__ == "__main__":
    # Look for specific hashes to append
    selected_hashes = [
        "3829ef90861aacbbcbe477db12e91f43e8cda86a7e4f0bbdf19972c97ca01341",           
        "2c74a967c4d5d66110edf9343b9299258856a0a019e9e155c7d74566e84c8040",        
        "31b2004e6b5958e958a0af48816a02dff3dc0d090e962a47b379e6d18c2a0b59",        
        "44689faec8191645e50ba7246b00e17dfa6790664c09aee2f34e38138c35ff94",
        "7759822e4efaf49c34b4658ed07ca050d32398aa26788eecd9c1c807241180b4",
        "685a73ffc667708bc377fa24284d6a4533d027ea3b02838b79982b924d16c0fb",       
        "7fa72de2b26aa5e75e34048493a6eaf19780cbf6a093f85f25da6faad1a93433",        
        "72e6965e3cb17dc3a8402f9479e0ec90a8d82b14c53d6680efa3ac168d169a54",
        "395b5f9263f8ea1354a3e7e0ad62075610cc92eff89faaafd11b31ab929f8cc1",
        "d3b3dcc34b44bf5999fb06ef9601e30e814a20b2e64e93b53309c9cf1f6ad2f4",
        "c4ba4e5fbb211c88774b8e1fc5232ce0764c5a13a92c6e131d7116756c756d22",
        "ef25ec3b20b6d646ad07ced93db6efed445f3755642abdd821d91ece78b650a7",
        "a7221f3174c674efaa800b2e5e7a67bb5dfba2187e635716da464433ee5506b4",
        "767e1343ef8251aa6f5a27ea4f6a19456ff1d1838c54f845e08d58bda8706e05",
        "c370ae4721a09226f93545e791702a8bc0f1e9585f0ebc6c2ac687bed5e19f22",
        "5424400038493d0b3a2bf0a2ca264f4883c5240a99f3402f7dd9d08cddf39448",
        "18fcc2c99a4969ce55b5aa8c21005a2c629255e3eec9e9e32e6b1000cc79d51e",
        "1dccfa7d4c53dc419a8c8949eb5bfb5548e6daa2eb357198ab04afacf90efd4f",
        "51892b33d412acc945464c934f61827b95c5206c18801548a73c91c3f3cac89d",
        "a8f92c3306be15628f0ef89824908b2c7dbf490340d0efb9d874bb2ffb729552",
        "560f1b615eb82e1634ea960be8d41bd7badaf7a19fa7d7c3a8ea746df30e6c04",
        "9c7cba9787d39a8c97bd7c920edfa23134ee5bd2a959754e2b9108c74aac77e3",
        "9cd77be652343a6485af8b73127f6c62981fedd82ac97334aff8225b9ce34203",
        "96cceede683029d5c4b6dfd9d3ad4f2f5953ecde6ef5dca89af3f77aedc1a434",
        "ecaeb11606c8cca560d9b017618068b2464be6c827e4be9f35a48996d07d9a0d",
        "c020dd92c9ed049d903cbe475a41cc6d8fed3331fcd7462197b74ba00f249fd4",
        "9d3147c4a97c41159df23ff95ca9dd903de3148e2d7b6dc8cf9d24c5b6ea07ef",
        "61b09b93bda0bf10fe90a1e891e0fb46949a99586348fb9eeb8727154d79978c",
        "f60130724da4c7d95892558d9c731a6da16c838da7271a93b750b88e8d6883e2",
        "37b25d45030f9ba7b80bb5586008d8cd1b3427e9939ee9a23f3e758cd03de9d8",
        "c776347468dea312c357ecc7a6f87877d752240a380a83d99ef443b52340c766",
        "cf04d17b18966cda188beed6ea9406a8b43864e1f5304dccabef8f304a18e900",
        "22753173ed34b884483e0f73ffe1c83e978f2e643e3f2a52ff69062aa683690d",
        "74f2f2349f0abd2219abb1283a2995422a5f02cc4e27e63a67416f6e43565d8d",
        "b0210f16f335ad77f5590a88040ad526f332ce2c46be534ad5989a1487fef147",
        "1e04b27a8025dfff179089e48997ebad22ae7597815dbb581cba11136c766f50",
        "fc34997d10bc9e5436eb04de3bb7bc3a1c18060ce143e3f40de57475e852076f",
        "1ce53ba93b59fb8f35af80caf14dcde97ef816e5da74f374a1d6b816850da8be",
        "529c3cd284f7fdbe5c503e3542fb4164af12fef87a8313c3d6952bbc5ab4cb78",
        "a71ab7984d91ff926ca5ac611f499014f64392b16c064a2a8d4aaabfb8422828",
        "837f0de04614fb677190cc8c3a7e563d81791fbcc540f6acd31ac18d8fc50785",
        "c01b797629b2a06526b8f57a36a15eb8710fc92cc1a574287c211c02dd24e42d",
        "0fce93b721bd5e26b7097d1c4bc50ead3d99dc83b6549d477ff763b342c712ce",
        "74d28887e1e4de6f1e38d730fc813420198a5b134ea96fe7fa4bad08384642d8",
        "1c87617a213d08eb4ed1693a7b1beef80df0bceafc77890e85355f314e28b181",
        "7df31a1080900701bb746f66751b06270119a4e761999caf4569fd8f33489fb7",
        "1f5247cd6515d69a180370e6d2ec572aa4aa98905b86923517469c69bb4ffa49",
        "bf67ab0c7bc1f16c763a8c36c6778c84923eadc96e41e5e11e0baa840c3c0314",
        "940d534891d0a8df85086382f3fc20030bf50dc7e1bdcc92921ad4c7fef87ccb",
        "7c1d3329fd5935311bc367b013b0b7106cce9495315fd9fe6d11efd649506dc7",
        "aff37f18febfa016fce3998591441185760dde10b78206673696102f4a3cbec2",
        "48b210b9a59e81ff5a93cfc6e4f351d9450d9362ccd76283527cedf14b70503c",
        "6deaa13987c8cbf4a75200bc7f729bdbe70ecbccc2d79ccc70092c423652c977",
        "112197603146e6ff54fb097e97d1bede1122e51621f59fa95b1b5722a12d12a7",
        "25f33e510e136f3bfe57a8521f5b32b3659e5183a39e005b8a8e01620b0ad5b4",
        "014bf5273a84980cec888d49641772bdc18ea34dd09788798cfa89132e13d608",
        "76df052b8dfa4db360e00b21740d0ef7efaf73dcd740a8852c5d534432c3374c",
        "6b0daf2935959baff2fa8dec44b19441993354484162b51310b1626acf3a52a3",
        "2a0952fe6dc81cfb6d127b55efd78f72371afb9d442befeb2f3eea65cd94df82",
        "9594a885826f8f52cea53de0b05fff236e74c9e91462971bb2d46e121304b6fb",
        "c1252e5a45b164582eba178d2f60fb5f8c423cfecbbc6e611ffe90a654dca59c",
        "9ef52cdfb99665b82a5fd1a6aad5d744a27f5a77a2764d82d6fb265095c0549e",
        "833104e69e340902a40c0e532289cafdd723b75e8d6851773fc9de558b9d4886",
        "3350cca35b63d9173de3173c10490e5ba696ed9936fd069be0c3693bd5451e0f",
        "1a0d47c7cb5d619a7533eb25716b1b119d57696e10c436a78e08464a5409080b",
        "3c0eb5946742e593b96da1c36b1a7f116c470433a8769de1f883fdb94e0f464a",
        "710018d13787d7fe7c49f5f18ab9e11d615ccebcf90b5642c8185ec1943e780c",
        "033fb3ae386e604c83c5750b6895224d2b1a8fb0f94076e3efd7ab883b4ccf87",
        "a17216ff0f39964c2a3f812b78ce5f442907d661bd25c26a886bce0a6f9d96d2",
        "2379cc5e2d6ce92c2bc4ce2afccb31e700dfaa49af79ebe27330d9a3212ff4bb",
        "52c9e1fc8b4c746dfc3d999ae4ade42454655354d632055a44d2930bca1881a0",
        "5abbe25337918eefac49daedfeda0a9441fe21248fd3b8d72eafd7b57339bdcc",
        "202f3e2d316657396b997db8295487a7aa71146f571eee62e4b8968533e8e975",
        "6b6c76eee26dbded1c596e3c0d7327cfd890b790b2c5a3fd69f50e21aa2642fd",
        "e28b6a8ec13282ed9a4e1b04accfc2768ffa72ce1aef491be99faf3befdfa55e",
        "d7f9c8ef205d6d93e57294beb062aa29158109f157ab21591e5bd7110f640cf4",
        "c166f8f5beef571f95c54609c6400a26d8d21b565385785ef3c38fcc853199bd",
        "dd119a55240276bd852e1c1776817bdb76a2871b915779eb2b32295d1fd20964",
        "eced31426c3ea952e0ef3e37d38b46747612635d53529d05970d47cd7511dbe3",
        "7bc3da943fab1caad3a6a0c15f508158e3cd63748ae94a172eae39d7bf16a363",
        "546b2f90ef22e1568c7860bc0f2139c617151012294762d01cd3707ce34c5022",
        "ccac26ebea70f03b8a92c361d26cd46ee914d49a3cd4b022cc204830b48d898a",
        "6aa208949eaba411582782522526dac5a490ca11595019bff9929c614dac6530",
        "81e08f03e2a5333ccf3da2acd214bda1e63d3d0e79f58874fb3d7a75dd1f813a",
        "121da87b46adc256412eec852b7dcd90e52f3694ab9b23147ab82b879b24532b",
        
        
        "9656dc63dc1f824623d59bb272e7bc5a6f6843efac8c8e532610627a019b8ede",
        "d93b04e079659bab839417bd9dbf2c09f68177fa329d98f4c3a8b9d1286121b1",
        "7aa10f0b435f09b38737eafb102a385f69c33b429469a63ed9bf4cbd6f798fda",
        "463cedbffd1ff08b5e1ffb9653aec09f61781f69844155cfc81f65e37c98254f",
        "f260305e1017e29d07c3ccd78fa18257affbe38da2451ea13ed47ae4ca56c170",
        "baa64df56e4813d00374c0fa8e3685eac83d0cc8c961527f097d1f6a50a66346",
        "e14e0b922a99a13906f47d9ce0d6379c906fdee44962f00a8452b72228d5fdd9",
        "aa0342f022bbfce8389b275997e2f7a09fe7c0e7ac1a54a263967e77b56e0ffd",
        "60b7007347a717671662a3d09fd726b174b8c10b84f78ac11427e5857b065ff6",
        "3ec939ddc418f2998f7bb53315e5ae126bb315e905334d07a11b68446a4063ad",
        "40da687894e0e9688953159fa2f89a7bc39ca495d85507b35604621a04af57f8",
        "3a0c26a8ea0e781e00b3a900412e0fb6dbbd4e1124bc3f69d49f8d7aa555a7a4",
        "8a7fc2f8c36b76ff95ee67f27fab919705db1890d66f2f2ef4cd3ce2922095e6",
        "1d64de03b5fc69aef66ddd3c1ec434e28ab9d50c569a9c1fbb627d60d6d16806",
        "7f563e71a945a24efbded5ca157883f18135a0daecd16ae117be9da9f6206de4"
    
    ]       
    append_accurate_predictions(selected_hashes)
