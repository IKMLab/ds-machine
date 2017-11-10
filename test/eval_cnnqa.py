from DSMachine.dataset import DataTransformer
from DSMachine.evaluation.evaluator import QAEvaluator
from DSMachine.models.CNNQA import ConvolutionDiscriminator


def main():
    data_transformer = DataTransformer(path='dataset/Gossiping-QA-Dataset.txt', min_length=5)
    qa_discriminator = ConvolutionDiscriminator(vocab_size=data_transformer.vocab_size,
                                                embedding_size=200, kernel_sizes=(2, 3, 4), kernel_num=48,
                                                hidden_size=144, out_size=2, conv_over_qa=False, residual=True).cuda()

    evaluator = QAEvaluator(qa_discriminator, checkpoint_path='checkpoint/CNNQA_WITH_RESIDUAL.pt')
    evaluator.interactive_mode(topk=5)


if __name__ == "__main__":
    main()