from groundedsem import GSEM


gsem = GSEM()

print(gsem.device)

gsem.detect("assets/SEM/2_1_5.00kV_0.34nA_1.35nm_123_ETD_3.00µs_yes_200000×.tif")

# gsem.dino_sam_segment("assets/SEM/SEM_1.png")