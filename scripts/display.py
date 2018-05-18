import pandas as pd

out_hori = pd.read_table('out_HORIZONTAL_SD_LSTM.tsv', sep=' ')
out_md = pd.read_table('out_MD_LSTM.tsv', sep=' ')
out_snake = pd.read_table('out_SNAKE_SD_LSTM.tsv', sep=' ')

print(out_hori.shape)
print(out_md.shape)
print(out_snake.shape)

out = pd.DataFrame(pd.concat([out_hori, out_md, out_snake], axis=1))
print(out.shape)

print(out.columns)

out.to_csv('out.csv', index=False)
