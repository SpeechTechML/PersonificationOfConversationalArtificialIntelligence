PREFIX="/mnt/storage/logunov/data"

# DailyDialog
python3 -m data_parsing.daily_dialog -i "${PREFIX}/ijcnlp_dailydialog/train" -o "${PREFIX}/nirma_2022/daily_dialog/train.csv"
python3 -m data_parsing.daily_dialog -i "${PREFIX}/ijcnlp_dailydialog/test" -o "${PREFIX}/nirma_2022/daily_dialog/test.csv"
python3 -m data_parsing.daily_dialog -i "${PREFIX}/ijcnlp_dailydialog/validation" -o "${PREFIX}/nirma_2022/daily_dialog/val.csv"

# MELD
python3 -m data_parsing.meld -i "${PREFIX}/MELD.Raw/train_sent_emo.csv" -o "${PREFIX}/nirma_2022/meld/train.csv"
python3 -m data_parsing.meld -i "${PREFIX}/MELD.Raw/test_sent_emo.csv" -o "${PREFIX}/nirma_2022/meld/test.csv"
python3 -m data_parsing.meld -i "${PREFIX}/MELD.Raw/dev_sent_emo.csv" -o "${PREFIX}/nirma_2022/meld/dev.csv"

# IEMOCAP
# ...