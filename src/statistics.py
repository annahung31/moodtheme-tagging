import argparse
import pandas as pd
import numpy as np
import os
import commons
import util


def get_statistics(category, tracks, tags):
    data = []
    total = {'track': set(), 'artist': set(), 'album': set()}
    for tag, tag_tracks in tags[category].items():
        row = [tag]

        for collection in ['artist', 'album']:
            collection_ids = {tracks[track_id][collection + "_id"] for track_id in tag_tracks}
            total[collection] |= collection_ids
            row.append(len(collection_ids))

        total['track'] |= tag_tracks
        row.append(len(tag_tracks))
        data.append(row)

    data = pd.DataFrame(data, columns=['tag', 'artists', 'albums', 'tracks'])
    data = data.sort_values(by=['artists', 'albums', 'tracks', 'tag'], ascending=[False, False, False, True])
    data = data.reset_index(drop=True)

    total_stats = {category: len(collection_ids) for category, collection_ids in total.items()}
    return data, total_stats


def write_statistics(category, data, directory):
    data.to_csv(os.path.join(directory, category.replace('/', '_') + '.tsv'), sep='\t', index=False)


def compute_statistics(tracks, tags, directory):
    util.mkdir_p(directory)

    for category in tags:
        data, total = get_statistics(category, tracks, tags)
        write_statistics(category, data, directory)
        print('Total tags for {}: {} tags, {}'.format(category, len(data), total))


def compute_duration_stats(tracks):
    data = np.array([track['duration'] for track in tracks.values()])
    print('Mean: {}, median: {}'.format(data.mean(), np.median(data)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computes data statistics, such as number of unique tracks, albums '
                                                 'and artists per tag')
    parser.add_argument('tsv_file', help='TSV file with such columns: TRACK_ID, ARTIST_ID, ALBUM_ID, PATH, DURATION, '
                                         'TAGS')
    parser.add_argument('directory', help='Directory for computed statistics')
    args = parser.parse_args()

    tracks, tags, _ = commons.read_file(args.tsv_file)
    compute_duration_stats(tracks)
    compute_statistics(tracks, tags, args.directory)
