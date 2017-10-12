from dataset.data_helper import DataTransformer
from models.CNNQA import ConvolutionDiscriminator
from eval.evaluator import QAEvaluator

def main():
    data_transformer = DataTransformer(path='dataset/Gossiping-QA-Dataset.txt', min_length=5)
    qa_discriminator = ConvolutionDiscriminator(vocab_size=data_transformer.vocab_size,
                                                embedding_size=200, kernel_sizes=(2, 3, 4), kernel_num=48,
                                                hidden_size=144, out_size=2, conv_over_qa=False).cuda()
    evaluator = QAEvaluator(qa_discriminator)
    evaluator.interactive_mode()

if __name__ == "__main__":
    main()