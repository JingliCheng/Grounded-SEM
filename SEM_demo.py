from groundedsem import GSEM


gsem = GSEM()

print(gsem.device)

# gsem.detect("assets/SEM/2_1_5.00kV_0.34nA_1.35nm_123_ETD_3.00µs_yes_200000×.tif",
#             text_prompt='white circle', box_threshold = 0.05, text_threshold = 0.1)

# gsem.detect("assets/SEM/2_1_5.00kV_5.4pA_13.5nm_132_TLD_1.00µs_yes_20000×.tif",
#             text_prompt='white circle', box_threshold = 0.05, text_threshold = 0.1)

# gsem.detect("assets/SEM/2_1_5.00kV_0.34nA_5.40nm_117_ETD_1.00µs_100000×.tif",
#             text_prompt='white circle', box_threshold = 0.05, text_threshold = 0.1)

for i in range(1,14):
    gsem.detect(
        f"assets/SEM/SEM_{i}.png",
        text_prompt='particle', box_threshold = 0.1, text_threshold = 0.25
        )


# gsem.dino_sam_segment("assets/SEM/SEM_1.png")