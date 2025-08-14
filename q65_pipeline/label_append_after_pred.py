### Re-Label Based on Accurate Predictions From Scored Subgroup ###
# (Ran after the first prediction and never used again. Provided 62 additional sentences for labeling. Potential reduced number of matched sentences could be due to hash mismatches. 
# Did not use the full unga_wvs7_hashed_corpus.csv, but rather the top scored sentences from the WVC corpus.)

import pandas as pd

def append_accurate_predictions(
    selected_hashes,
    top_scored_path="../top_scored_sentences.csv",
    labeled_path="Q65_mmr_selected_labeled.csv"
):
    # Clean hashes to avoid whitespace issues
    selected_hashes = [h.strip() for h in selected_hashes]

    # Load top scored sentences
    top_df = pd.read_csv(top_scored_path)
    top_df['embedding_hash'] = top_df['embedding_hash'].str.strip()

    # Find matching rows in top scored sentences for all selected hashes
    matched_rows = top_df[top_df['embedding_hash'].isin(selected_hashes)].copy()

    if matched_rows.empty:
        print("No matching rows found in top_scored_sentences.csv for the selected hashes.")
        return

    matched_rows['match'] = 1

    # Load existing labeled data
    labeled_df = pd.read_csv(labeled_path)
    if 'match' not in labeled_df.columns:
        labeled_df['match'] = 0
    labeled_df['embedding_hash'] = labeled_df['embedding_hash'].str.strip()

    # Append and drop duplicates (keep existing first)
    combined_df = pd.concat([labeled_df, matched_rows], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset='embedding_hash', keep='first')

    combined_df.to_csv(labeled_path, index=False)
    print(f"Appended {len(matched_rows)} rows to labeled data and saved to '{labeled_path}'.")

if __name__ == "__main__":
    # Look for specific hashes to append
    selected_hashes = [
    "adbb926eb87903ec1f57a5e9adc87cb2202609531b1f6aea7a90d8117faddf95",
    "e8ed81c8a88b73f87cecf69d5e5159e63181d7b893ad3d9f3c9e29b225653de7",
    "7b6d633686655e3f8477450635e095c36773eaa999627f1d6484be3c2775cc87",
    "c0c845850c6495bef40dfb28d77cb5a7f8a037814771ff88595c77a6bf7e6a70",
    "c8d4b180ca7c92b06fd7d027ce04736089b189768e7b5bded7b74b810629f3fc",
    "7be98c3d6599cec83cee436e8da04b53648b1acdd87bfa8d3da6b9521bc8f378",
    "f5ab12afe24b8358fb5d30458daacc268aa3a41c7adf983b57c644b0649b1c89",
    "6da0351d25286c5d8aa42ed2f964324040c2ce1a7b22929e28c4010fc0d1d4d4",
    "bc6eab3ae25e9fca0af2a6b43937b4a485fafff982ae712e4d7ee3d2b8c6d2ba",
    "c7a831a9c527061b751ed6e4e20018528dd5cdd87b5fa40c476b1e61855230c7",
    "7b4180c81ee053a250d3bcca2c715e26724e4751066c08e00b817f7077f59b40",
    "65600eed274dbb6a29e1553a36301e86aa9472b8ae424d32941764125ba78c98",
    "302a19e96d07dd44ff42edf434ea7b048b743d97902c2471dcbb0e0c8e122b15",
    "a0e96f92429cfde1e6b61d5f1dd85257d8aa6f7bcb8adf143a1942fc11a95d99",
    "aa47e1abe73a4f993a1678cfd7f7796156b79827def2aee032c238736856493d",
    "b5b1d360dc2a8b6b45486f462a8369d8b09eff5d3e9adcd0b0833837f86e4437",
    "e610821f80a43cde4d10fd684be07ae8bfba228db52b6a8dff4e14101d911dee",
    "5c137cec981373b58409b2625ac2337a7dea72ad9fa7ef04a621b2294e32cc56",
    "5c137cec981373b58409b2625ac2337a7dea72ad9fa7ef04a621b2294e32cc56",
    "a726ca45c43200bbaeaafee001dc084de106349ff3fd2c7a2b7d905534e4a61e",
    "44097af97385a0f3e975a64013767ad17f0f1efa089d00a40127b4863322c71a",
    "17bb951b87ac6f3639a137675b1e53170bd0a5cf54114527935e431168ecb909",
    "e81e9ecea75cad112b35b32bdcccbe0f63c3d2b1952b8426ab164ac666628eff",
    "225d3d00ac7f12a0cbe21ab0947307ace5f67799d7e0c2a8a46cdbeab5571f8b",
    "353cd9e5202def95fd8600ae04564a850113c2a301cfb28cac41e33e996459bd",
    "3982f26ab2cb234db05d155b505693082ebfdacb332928b16081ae2a5797464c",
    "4d8e1c38ed5372756dfdabfcb218973af1ae5aa6f7f8c0bcafdad7e1122eb611",
    "37f2dbba79aa777cbedc1f6da141ab8e54d027ad5558f5e43c26e663b5245cd8",
    "b9278c4a7037d165a668f7994cb78b11853597d69ae54ca73da7201277606a71",
    "95a08eba4c92df26cb145433afbefe35bab75f6eaa1bea81e58a9a3338c830ce",
    
    "9a8162290692e348b1b714d30821a101383e73fb9174ad0cbfa0505fa416ec02",
    "86a4d90da1befccc981ed024bf7fdd36bc293465da59844cd677d174d03bfa7e",
    "cdfd5bb3e01808a90a4362d7f9c952b14c8ec8ec3bd8a8bae6fd5a458e5feb8d",
    "85c3a606c3336d51b87bb9e63e1ff6ecf053cb8943a5620e2b95e433bd20c37e",
    "8945da9b09f4ed354a13e0250224bd79c1fcb5ab989ecb2d6cd68137f2b28322",
    "e45f1df79900d253faf8cb1c2d4df72a9bc845f46088cb9c04db9b5688e0fc35",
    "84ff6f141e80281e762d894840fe0fa7272d4b8a3ab3a444e9d4fba8d8ed9f53",
    "6ef953c5e1fc56d976a2c03502efbb667fbfdc39a9d1cecb1bb423971b34f314",
    "dcedc17202437d24c4aef4d0ba246f474e32c60227bd9a2846c87388aa3d8d7f",
    "c8aea27755c32d3eb837288730b2107b58ef23126db08deee04e1e3292f44b67",
    "6a91165e97e97319e48157462e44a479d232670d7667194e2c6a8e932a5cd01f",
    "c68dee2dbe797947c7f77387f22135868d9c13ffb31f95be0eeda0fcebc27c35",
    
    "4aaca94df68328f3c4e0186dd9e7da110c48e583fe8129e37572621918b28869",
    "695e0242b616ea23fd5d0081bd1be4bf8c2b319d5e6610c96b9fffbde316a905",
    "dde690f7453478a31709f9bccfa07c1430be673c640a2c5cb4c04f86c7f34b2a",
    "a667df60feae7dacda690021285af42ec05cf0fb04191551e80e0a3e48c127ec",
    "a5f762567f5f91bcfa75193d9dd62f1faa4543ec9d572425935e67eebfe92c02",
    "ccc7d161c60aa352cb1220487411b1507daf8eef0cd7ee36ff6ff3a00a03f768",
    "da7d0263103f00fbf6d536f07e05eb5f2c531d97c56de1a485c4037f5deebbc0",
    "bf4b81c38cfd65cc29ce15a26f3d21472b9417175cc8f8e00dde0a7ed5d19dee",
    "a5228ebbd0994f287854ab74bed0628422ed191ebe2a7e093c9797108ec20c2f",
    "73c46f984d6556b674e0dc78b4483f34d31c15cadba87face5e46e4b9e3aa16d",
    "468a7f3636387a9cff8cbf4f5f56727ee4f4a3b9df0801291097ca34c0716a5f",
    
    "e775f194ed5052592c58a6234be1e0c11b4a6680cc644227a3a96d5869bbdef8",
    "0ab261200870b1121dcbead1645c7dd47ff036f3a7f19662f09e0473f30e9a75",
    "e79bfc6a7850a6892bfd8d3fdea4476d1b282fab4c1e8c55fa64fa10ef407655",
    "55b8526ae4f5cb5e5230bb39f14daeea5bc3ab14548e8eddbe1a9b3d2fb260f1",
    "4f552b185d375aa6914f737b5741219f2af1c5cbdf871a7bfed7b3533ca87952",
    "c869334ff4f575ef465b9f9cc225811896b2fa0d1051914ed3b04606c8501cd6",
    "2322edd8c8593aad8e2fb9997b65e3b733daef6f334c7f3ad549a7862589278e",
    "ce6007b719ff661a9b5fbd81c0defa550aacfb7c0562af0d3372731769660447",
    "12733d88f223aef4edfc91a2e51e688b9f94f567f01d683e56626d994715b5b4",
    "50cd37c84df9853d7b5411cd47abde7d9502b95a49091445ee21eab772aa6458",
    "bdafecb671e0624073f057a2c69cb2f215a351994623bfa1b625eb40ad0123ff",
    "66c66de4291f8319d0961e43357d3675963c4c72e3ef564b537e5aaf1254c530",
    "e3057db7309d65e1a27b80c00c8f4a4b8a6bd0a73dab545b6cb0a3245db92b2c",
    "5f34dcb154211e86d2e0d7130b66f3e74419aca31c71182bbe3bd9d732c0fd8b",
    "aa18b5b14cbece665ed3df77bdec9b55b5b9431fe918a23208181d35db2f3bcc",
    "69d87e56ed2b864efc8fdfeb3affb56895aecfdfd464eb04e7bb769fae4e23b0",
    "66411eb92a4ddb490f1ecdafbfba98b55dc151c8e5a6612935e3997b5a72904c",
    "6ecccbb357d42132ba555c42b0472bdbf1965d2fa04f61b90e369431582da641",
    "8e95384ff1ea7ebc7edcee64ee55a1e81eeaef882c8935db67728e917977c996",
    "c848adccbf69af4459dbbe506661d77a19a6c9dc259ea5814ca73e17c2453d0c",
    "01c5bab3e2fa66d6d985ab24cb769a5818c0f96bd4d96290c5474f647539caa4",
    "4059384dcb60e84a1ad563fffb346665873f420e8cade735924b3c7e2492103a",
    "f1fc2be1a1af07bc075eb87e56cbaa00e9ad2bd79220366428f67154e5bac208",
    "096fb8eceaad11642864b80317d158ff13251b4d2141bf889a1577d4ff31a0e5",
    "616aebc52f80a4d8fa4fedf6e3944bbd8fe1ea4b03080a4d04587c467a0747a8",
    "4bc96db96391c64c3a5b9c64a06fff3435f8ca276e143953962a1a83e0ff5b44",
    "32cb93ee4326a253c66cdfbea0fe33c5b37674476711e5411178bd2d6342d86f",
    "ca97330a622e31fa5555293405a5fafbb65e5e0572be2a96e2329b28fc0fb33d",
    "5f27f9792fd09de568360edb9122446a208c722a0d3c949c57a4b6afafa6ce27",
    "bb2b1f6d5b639182c415e14b34f886117709700c81f643bcfb17a21ee82cc245",
    "0a096ad03836a33eb20bae7a34d5ad908c93d2c6f2bed5ed103971f1756bca4a",
    "29127344d18ee3b24d6832ae4dec560928268ac0f84e0aa6f35b791bef121a26",
    "3b3b9885be132a5aedeb0801c026e60a24b240cd1c3e39c48a8a6af55dae5151",
    "f361c62c75e7470174ae73eda01de35ec3304604a8a9327b92e73fd4aba18e35",
    "bfa9900c858219bfaaeb89c015310d8c8e4a048d7f13ce164bd76d7325370ad1",
    "7dcc806555ee0ef37466dfc16708c547aa14ebe4ad50d355cf1e1c8f0e54f793"
    
    ]       
    append_accurate_predictions(selected_hashes)
