from DSMachine.dataset.data_helper import DataTransformer
from DSMachine.evaluation.evaluator import QAEvaluator
from DSMachine.models.CNNQA import ConvolutionDiscriminator


def main():
    data_transformer = DataTransformer(path='dataset/Gossiping/Gossiping-QA-Dataset.txt', min_length=5)
    qa_discriminator = ConvolutionDiscriminator(vocab_size=data_transformer.vocab_size,
                                                embedding_size=300, kernel_sizes=(1, 2, 3,), kernel_num=72,
                                                hidden_size=500, out_size=2, conv_over_qa=False, residual=False).cuda()

    evaluator = QAEvaluator(qa_discriminator, data_transformer, checkpoint_path='checkpoint/CNNQA_WITH_RESIDUAL_POS_50_CLEAN_STOPWORDs.pt')
    evaluator.interactive_mode(topk=10)


if __name__ == "__main__":
    main()