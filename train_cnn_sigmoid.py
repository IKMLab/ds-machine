import torch.nn as nn

from DSMachine.dataset.data_helper import PTTSentimentDataTransformer
from DSMachine.sentiment.sentiment_classify import SigmoidSentimentClassifier
from DSMachine.trainer.supervised_trainer import ClassifierTrainer


def main():
    # data_transformer = SentimentDataTransformer(dataset_path='dataset/Weibo/weibo_with_sentiment_tags.data',
    #                                             emote_meta_path='dataset/Weibo/emote/sentiment_class.txt',
    #                                             use_cuda=True,
    #                                             char_based=False)

    data_transformer = PTTSentimentDataTransformer(dataset_path='dataset/OpenDataShare/',
                                                use_cuda=True,
                                                char_based=True,
                                                sigmoid=True)

    classifier = SigmoidSentimentClassifier(vocab_size=data_transformer.vocab_size, embedding_size=512,
                                        hidden_size=768, kernel_sizes=(1,2,3), kernel_num=72)
    classifier = classifier.cuda()
    trainer = ClassifierTrainer(data_transformer ,classifier, learning_rate=1e-3, checkpoint_path='checkpoint/clipped.pkt', loss=nn.BCELoss())
    trainer.train(epochs=50000, batch_size=512, pretrained=False, binary=True, sigmoid=True)

if __name__ == '__main__':
    main()
