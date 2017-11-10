from DSMachine.dataset.data_helper import SentimentDataTransformer
from DSMachine.sentiment.sentiment_classify import CNNSentimentClassifier
from DSMachine.trainer.supervised_trainer import ClassifierTrainer

def main():
    data_transformer = SentimentDataTransformer(dataset_path='dataset/Conversation/conservation.txt',
                                                emote_meta_path='dataset/Weibo/emote/sentiment_class.txt',
                                                use_cuda=True,
                                                char_based=True)

    classifier = CNNSentimentClassifier(vocab_size=data_transformer.vocab_size, embedding_size=256,
                                        hidden_size=512, output_size=2,
                                        kernel_sizes=(1,2,3,5), kernel_num=64)
    classifier = classifier.cuda()
    trainer = ClassifierTrainer(data_transformer ,classifier, checkpoint_path='checkpoint/Conversation3MAXPOOLING.pt')
    trainer.train(epochs=10000, batch_size=128, pretrained=False)

if __name__ == '__main__':
    main()