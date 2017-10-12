from dataset.data_helper import DataTransformer
from models.CNNQA import ConvolutionDiscriminator
from trainer.supervised_trainer import QATrainer


def main():
    # TODO store all configs in a file
    print("Loading the dataset...")
    data_transformer = DataTransformer(path='dataset/Gossiping-QA-Dataset.txt', min_length=5)
    qa_discriminator = ConvolutionDiscriminator(vocab_size=data_transformer.vocab_size,
                                                embedding_size=200, kernel_sizes=(2, 3, 4), kernel_num=48,
                                                hidden_size=144, out_size=2, conv_over_qa=False).cuda()
    trainer = QATrainer(data_transformer, model=qa_discriminator)
    print("Training the CNN-QA-Discriminator")
    trainer.train(epochs=10)


if __name__ == "__main__":
    main()