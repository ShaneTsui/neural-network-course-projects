from applications.music_generator import MusicGenerator

conf_path = './conf/LSTM.yaml'
music_generator = MusicGenerator(conf_path)
music_generator.train()
music_generator.test()