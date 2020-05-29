# summarization - project report
*Project for NLP course by Huawei, spring 2020. Russian text summarization*

*Done by Alexander Kazakov. Date of report 29.05.2020*

### Code is here
https://github.com/AlexanderKazakov/rubart

### Previous work
There is abstraction-based summarization and extraction-based summarization. 
See [wikipedia](https://en.wikipedia.org/wiki/Automatic_summarization) for details.
In this work, abstraction summarization task is studied.

Text summarization problem already has many approaches listed [here](http://nlpprogress.com/english/summarization.html) 
with ROUGE metrics reported for different tasks and datasets.  

There are two public datasets for abstractive summarization in the russian language:
- [Lenta.Ru-News-Dataset](https://github.com/yutkin/Lenta.Ru-News-Dataset)
- [RossiyaSegodnya/ria_news_dataset](https://github.com/RossiyaSegodnya/ria_news_dataset) 
- There is also a non-public dataset from sports.ru: matches broadcasts as texts and short articles 
about these matches as summarizations. However, this dataset is much smaller and is not very consistent

both of them contain texts of news articles and their titles. 
Generation of the title from an article text can be considered as a summarization task.

Today, the number of works devoted to summarization of the russian texts increases:
- [Headline Generation Shared Task on Dialogue’2019](http://www.dialog-21.ru/media/4661/camerareadysubmission-157.pdf)
- [Importance of Copying Mechanism for News Headline Generation](https://www.researchgate.net/publication/332655282_Importance_of_Copying_Mechanism_for_News_Headline_Generation)
- etc

Most of the modern works make use of the popular 'transformer-like' architectures
([original transformer](https://arxiv.org/abs/1706.03762), [BERT](https://arxiv.org/abs/1810.04805), 
[GPT](), [BART](https://arxiv.org/abs/1910.13461)), which make gain of unsupervised pretraining

### Our approach
Following the recent approaches, this work adopts [BART](https://arxiv.org/abs/1910.13461) architecture to the 
task of abstract summarization of russian texts.

The only publicly available transformer models pretrained on russian texts 
are RuBERT models by [DeepPalov](http://docs.deeppavlov.ai/en/master/index.html).
These models are available as part of [Huggingface transformers library](https://huggingface.co/).

The idea of that work is to slightly modify the [original BART architecture](https://huggingface.co/transformers/model_doc/bart.html) 
in order to be able to reuse weights of [DeepPavlov's RuBERT](https://huggingface.co/DeepPavlov/rubert-base-cased) 
in the embeddings and encoder layers of BART.

Modifications [include](https://github.com/AlexanderKazakov/rubart/blob/master/summarization/load_rubert_to_bart_encoder.py):
- Changing some dimensions and parameters of the model (hidden size, etc)
- Incorporating token-type embeddings (dummy all-zeros) into BART model
- Manual weights binding, as codes of these two models are slightly different

Embeddings are shared between encoder, decoder and output classifier. 
As embeddings are from the pretrained model, the tokenizer (WordPiece) is also borrowed from 
the DeepPavlov's RuBERT model.

So, the only initially unoptimized parameters of the model are decoder layers.

### [Training](https://github.com/AlexanderKazakov/rubart/blob/master/summarization/summarization.py)
The model was trained on Lenta.Ru-News-Dataset as it is smaller. 

Tokenized input texts were truncated by max length 256, tokenized titles -- by max length 24.

Firstly, just model decoder's layers were trained for 3 epochs while other weights are frozen.
Then, all model's weights were trained for 5 epochs with reduced learning rate.

Also, after training the model on Lenta dataset, weights of this model were used to initialize the model
for training on sports.ru dataset. 
For sports.ru dataset, both texts and summaries were truncated by length 256. 
And texts were truncated from the end, not from the start of the text (is is better for summarization 
to see the end of a match, not the beginning) 

### Results
Results were obtained using top-4 beam-search decoding
####Lenta.Ru-News-Dataset
The average rouge results on Lenta dataset (validation part):

| r1-f | r1-p | r1-r | r2-f | r2-p | r2-r | rl-f | rl-p | rl-f | (r1-f + r2-f + rl-f) / 3 |
|------|------|------|------|------|------|------|------|------|--------------------------|
| 37.7 | 39.4 | 37.3 | 21.1 | 22.1 | 20.9 | 36.6 | 37.9 | 35.7 | 31.8                     |

Results do not show improvements over [Headline Generation Shared Task on Dialogue’2019](http://www.dialog-21.ru/media/4661/camerareadysubmission-157.pdf). 
However, manual check of the results on the validation set show reasonable quality of summarization:

| orig/pred | Title  |
|-----------|-----------------------------------------------------------------|
|  -------  | --------------------------------------------------------------- |
| orig      | США проследят за выборами президента Тайваня с двух авианосцев  |
| pred      | США направили к берегам Тайваня два авианосца                   |
|  -------  | --------------------------------------------------------------- |
| orig      | Французскую судью с 2000 года заставляли подсуживать канадцам   |
| pred      | Судья по фигурному катанию рассказала о давлении на судей в США |
|  -------  | --------------------------------------------------------------- |
| orig      | Два смертных приговора в США исполнили в один день впервые за 17 лет   |
| pred      | В США впервые с 2000 года казнили двух заключенных |
|  -------  | --------------------------------------------------------------- |
| orig      | Бекмамбетов доверит спасение Москвы американской туристке  |
| pred      | У Тимура Бекмамбетова появится актриса из " самого темного часа " |
|  -------  | --------------------------------------------------------------- |
| orig      | МИД Украины разрешил Лимонову съездить в Харьков  |
| pred      | Лимонову разрешили вернуться на Украину |
|  -------  | --------------------------------------------------------------- |
| orig      | Преступник запрыгнул в клетку ко львам и отделался укусом пальца  |
| pred      | Беглый преступник прыгнул в вольер ко львам и отделался укушенным пальцем |


####sports.ru dataset
The average rouge results on sports.ru dataset (validation part):

**sorry, it was lost :(**  
(but rl-f was about 21%)

#####Manual check:
***

- *orig:* 
В рамках регулярного чемпионата КХЛ СКА обыграл«Торпедо» (5:2). Sports.
- *pred:*
Сегодня в рамках регулярного чемпионата КХЛ « Торпедо » уступило СКА ( 5 : 2 ). Sports. Календарь Кубка ГагаринаСтатистика Кубка ГагаринаКалендарь Кубка Гагарина завершилась со счетом 5 : 0. Счет в серии : 2 - 2 - 1. ПРИМЕЧАНИЕ : время начала матчей – московское.

***

- *orig:* 
Сегодня в регулярном чемпионате КХЛ ЦСКА в Нижнем Новгороде проиграл «Торпедо». Sports. ru провел текстовую трансляцию этого матча.
- *pred:*
Сегодня в рамках регулярного чемпионата КХЛ московское « Динамо » обыграло ЦСКА ( 3 : 0 ). Sports. Счет в серии : 0 - 1. ПРИМЕЧАНИЕ : время начала матчей – московское. Календарь Кубка ГагаринаСтатистика Кубка ГагаринаКалендарь Кубка ГагаринаПервый матч – московское « Торпедо ».

***

- *orig:* 
В третьем периоде четвертого матча серии первого раунда плей-офф Кубка Гагарина между «Ак Барсом» и «Салаватом Юлаевым» (1:4) произошла драка между защитниками Райаном Уилсоном и Максимом Гончаровым. В итоге Уилсон получил 5 минут штрафа, а Гончаров – 2+5+10. Дисциплинарный штраф ему был выписан за то, что он бросил в Уилсона шлем.
- *pred:*
Главный тренер « Ак Барса » Сергей Мозякин прокомментировал победу над « Ак Барсом » ( 4 : 1 ) в четвертом матче серии плей - офф Кубка Гагарина. « В первом периоде мы играли очень хорошо, но в третьем периоде мы не смогли забить гол, но не забили. В третьем периоде у нас было много моментов, но у нас не было шансов забить, но мы не сумели забить. Мы не смогли сравнять счет в серии 2 : 1, а в серии – 1 - 3 - 3. В серии – 2 - 2 - 3 в серии : 1. Счет в серии 4 - 1 - 1. – 0. – 1. « Салават Юлаев » – 2 : 0. Как это было

***

- *orig:*
Главный тренер СКА Милош Ржига остался доволен крупной победой над &laquo;Спартаком&raquo; (8:1) в матче регулярного чемпионата КХЛ. Я рад такому результату. Правда, в первом периоде мы выступили очень плохо не могли раскататься, у хозяев были какие-то моменты. А через три минуты после начала второго периода матч практически закончился. Дальше мы играли в свой хоккей, а хозяева что-то развалились. Я рад, что в конце матча ребята не отпустили игру, а наоборот добавили и порадовались хоккею, приводит слова Ржиги официальный сайт Спартака.
- *pred:*
Главный тренер Спартака Олег Знарок после победы над СКА ( 7 : 8 ) в матче регулярного чемпионата КХЛ отметил, что команда играла хорошо. В первом периоде мы играли очень хорошо, но не смогли забить, но мы не забили гол, но в третьем периоде не забили. В целом мы играли хорошо, а в большинстве, не смогли сравнять счет, цитирует Билялетдинова официальный сайт СКА.


### Acknowledgments
Author is thankful to Valentin Malykh and other Huawei NLP course stuff for the great educational course!

