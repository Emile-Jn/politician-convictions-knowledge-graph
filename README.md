
## Repository structure

### `source`
 - [fetch-wikipedia-links.py](source/fetch-wikipedia-links.py) : Makes a SparQL query to Wikidata to create a CSV of all french politicians born from 1900 onwards, with their wikipedia links if available. Creates [french_politicians.csv](data/french_politicians.csv) and [french_politicians_with_wikipedia.csv](data/french_politicians_with_wikipedia.csv)
 - [scrape_wikipedia.py](source/scrape_wikipedia.py) : for each Wikipedia link, check if it contains a conviction keywords, if yes, save relevant section to [convictions.json](data/convictions.json)
 - [split_paragraphs.py](source/split_paragraphs.py) : split Wikipedia extracts from [convictions.json](data/convictions.json) into paragraphs of 300 words max, save to [convictions_split.json](data/convictions_split.json)
 - [info_extraction_wikipedia.py](source/info_extraction_wikipedia.py) : use CamberBERT model to assign conviction probabilities to paragraphs in [convictions_split.json](data/convictions_split.json), save to [conviction_probabilities.json](data/conviction_probabilities.json)
 - [extraction_analysis.py](source/extraction_analysis.py) : test HuggingFaceTB/SmolLM3-3B and meta-llama/Llama-3.1-8B-Instruct on Wikipedia extracts for better accuracy. Generate prompts as batches of 20 politicians for ChatGPT. Create files in [prompts/](prompts/)

### `data`
 - [french_politicians.csv](data/french_politicians.csv) : CSV of all french politicians born from 1900 onwards, with their French Wikipedia links if available. Created by [fetch-wikipedia-links.py](source/fetch-wikipedia-links.py)
 - [french_politicians_with_wikipedia.csv](data/french_politicians_with_wikipedia.csv) : CSV of all french politicians born from 1900 onwardswho have a wikipedia page. Created by [fetch-wikipedia-links.py](source/fetch-wikipedia-links.py)
 - [convictions.json](data/convictions.json) : JSON file containing politicians with conviction keywords found in their wikipedia page. Created by [scrape_wikipedia.py](source/scrape_wikipedia.py)
 - [convictions_split.json](data/convictions_split.json) : JSON file containing paragraphs of max 300 words split from [convictions.json](data/convictions.json). Created by [split_paragraphs.py](source/split_paragraphs.py)
 - [conviction_probabilities.json](data/conviction_probabilities.json) : JSON file containing conviction probabilities assigned to paragraphs in [convictions_split.json](data/convictions_split.json). Created by [info_extraction_wikipedia.py](source/info_extraction_wikipedia.py)
