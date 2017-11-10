from DSMachine.dataset.utils.processing.weibo import WeiboPreprocesser

def main():
    data_loader = WeiboPreprocesser(dataset_path='dataset/weibo.data')
    data_loader.separate_question_answer()
    data_loader.dump_tags(dump_file_name='emote_100.txt', lower_bound=100)

if __name__ == "__main__":
    main()