from DSMachine.dataset.data_helper import DataTransformer
from DSMachine.models.CNNQA import ConvolutionCosineSimilarity
from DSMachine.evaluation.evaluator import QAEvaluator

def main():
    data_transformer = DataTransformer(path='dataset/Gossiping-QA-Dataset.txt', min_length=6)
    conv_cosine_sim = ConvolutionCosineSimilarity(data_transformer.vocab_size, 256, kernel_sizes=(2,3,4), kernel_num=64, with_linear=False, eval_mode=True).cuda()
    evaluator = QAEvaluator(conv_cosine_sim, cosine_dist=True, checkpoint_path='checkpoint/CNNQA_WITH_COSINE_DIS_POS_55.pt')
    evaluator.interactive_mode(topk=15)

if __name__ == "__main__":
    main()
