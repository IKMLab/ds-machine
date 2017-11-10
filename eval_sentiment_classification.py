from DSMachine.dataset.data_helper import SentimentDataTransformer, PTTSentimentDataTransformer
from DSMachine.evaluation.evaluator import SentimentEvaluator, PTTSentimentEvaluator
from DSMachine.sentiment.sentiment_classify import CNNSentimentClassifier

def main():
    data_transformer = PTTSentimentDataTransformer(dataset_path='dataset/OpenDataShare/',
                                                use_cuda=True,
                                                char_based=True)

    classifier = CNNSentimentClassifier(vocab_size=data_transformer.vocab_size, embedding_size=512,
                                        hidden_size=768, output_size=2,
                                        kernel_sizes=(1,2,3), kernel_num=128)
    classifier = classifier.cuda()
    evaluator = PTTSentimentEvaluator(classifier, data_transformer, checkpoint_path='checkpoint/99acc.pkt')
    evaluator.interactive_mode()

if __name__ == '__main__':
    main()