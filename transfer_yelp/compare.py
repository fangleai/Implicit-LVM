
import logging

eval_path0 = './output/ARAE/25_output_decoder_0_from.txt'
eval_path1 = './output/ARAE/25_output_decoder_0_tran.txt'
eval_path2 = './output/ILVM/25_output_decoder_0_tran.txt'

# load sentences to evaluate on
with open(eval_path0, 'r') as f:
    lines0 = f.readlines()
with open(eval_path1, 'r') as f:
    lines1 = f.readlines()
with open(eval_path2, 'r') as f:
    lines2 = f.readlines()

logging.basicConfig(filename='compare0_1.txt', level=logging.INFO)

for i in range(len(lines0)):
    logging.info("--------------------------------------------")
    logging.info(lines0[i] + '\n')
    logging.info(lines1[i] + '\n')
    logging.info(lines2[i] + '\n')
