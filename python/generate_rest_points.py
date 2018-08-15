import mne
import numpy as np

from pathlib import Path
from MEGDataset import TEST_SUBJECTS, TEST_VG, TEST_PDK, TEST_PA

TESTS = [TEST_PA, TEST_PDK, TEST_VG]

LOAD_PATH = '/ais/clspace5/spoclab/children_megdata/Original/{0}/{0}_{1}.ds'
RESTING_DATA = '/ais/clspace5/spoclab/children_megdata/biggerset/resting/{0}/{1}.npy'


if __name__ == '__main__':

    bads = list()

    for subject in TEST_SUBJECTS:
        for test in TESTS:
            save_path = Path(RESTING_DATA.format(subject, test))
            if save_path.exists():
                print('{} {} already completed! Moving on.'.format(subject, test))
                continue
            else:
                save_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                raw = mne.io.read_raw_ctf(LOAD_PATH.format(subject, test), preload=True)
            except ValueError as e:
                print(e)
                continue
            events = mne.find_events(raw, mask=0x8, mask_type='and')

            # ensure good event conditions
            if events[2, 0] > 5000:
                bads.append(('Weird initial events sequence', subject, test))
                print('!'*20)
                print(*bads[-1])
                print('!'*20)
                continue
            elif (events[3, 0] - events[2, 0]) > 3.5 * raw.info['sfreq']:
                rest_t = events[2, 0] / raw.info['sfreq']
            elif (events[3, 0] - events[0, 0]) > 3.5 * raw.info['sfreq']:
                rest_t = events[0, 0] / raw.info['sfreq']
            else:
                bads.append(('No space for rest', subject, test))
                print('!'*20)
                print(*bads[-1])
                print('!'*20)
                continue

            raw = raw.pick_types(meg=True, ref_meg=False).filter(0.5, 100).crop(
                rest_t, rest_t + 3.5 - 1/raw.info['sfreq']).resample(200, npad='auto')

            np.save(str(save_path), raw[:][0].T)
            print('Successfully saved rest file.\n\n')

    print('Manual interventions for: ', bads)

