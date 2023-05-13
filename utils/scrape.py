import os
from icrawler.builtin import GoogleImageCrawler


targets = {
    'Rosy': {
        'keyword': 'cúc rossi',
        'amount': 730
    },
    'Calimerio': {
        'keyword': 'cúc calimero',
        'amount': 600
    },
    'Pingpong': {
        'keyword': 'cúc ping pong',
        'amount': 540
    },
    'Chrysanthemum': {
        'keyword': 'chrysanthemum flower',
        'amount': 214
    },
    'Hydrangeas': {
        'keyword': 'hoa tú cầu',
        'amount': 214
    },
}


if __name__ == '__main__':
    bot = GoogleImageCrawler()

    for flower_class, scrape_config in targets.items():
        save_dir = os.path.abspath(f'../scraping/{flower_class}')
        os.makedirs(save_dir, exist_ok=True)

        bot.set_storage({'root_dir': save_dir})
        bot.crawl(
            keyword=scrape_config['keyword'],
            max_num=scrape_config['amount'],
            file_idx_offset='auto'
        )
