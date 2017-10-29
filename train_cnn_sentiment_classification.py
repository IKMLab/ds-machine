from DSMachine.dataset.data_helper import SentimentDataTransformer
from DSMachine.sentiment.sentiment_classify import CNNSentimentClassifier
from DSMachine.trainer.supervised_trainer import ClassifierTrainer

def main():
    data_transformer = SentimentDataTransformer(dataset_path='dataset/Weibo/weibo_with_sentiment_tags.data',
                                                emote_meta_path='dataset/Weibo/emote/sentiment_class.txt',
                                                use_cuda=True,
                                                char_based=True)

    classifier = CNNSentimentClassifier(vocab_size=data_transformer.vocab_size, embedding_size=256,
                                        hidden_size=256, output_size=4,
                                        kernel_sizes=(1,2,3), kernel_num=64)
    classifier = classifier.cuda()
    trainer = ClassifierTrainer(data_transformer ,classifier)
    trainer.train(epochs=20, batch_size=128)

if __name__ == '__main__':
    main()