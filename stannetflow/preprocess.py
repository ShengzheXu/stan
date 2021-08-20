import configparser
from stannetflow.analyze_functions import analyze, extract, prepare_folders, recover_userlist_from_folder


def user_analysis():
  analyze()

def user_selection():
  config = configparser.ConfigParser()
  config.read('./ugr16_config.ini')
  # print({section: dict(config[section]) for section in config.sections()})
  user_list = config['DEFAULT']['userlist'].split(',')
  print('extracting:', user_list)
  prepare_folders()
  # recover_userlist_from_folder()
  extract(user_list)

def download_ugr16():
  print('Visit the following url to download april_week3.csv')  
  print('https://nesg.ugr.es/nesg-ugr16/april_week3.php')

def _prepare(folder='', output=''):
  if len(folder) and len(output):
    count = 0
    ntt = NetflowFormatTransformer()
    tft = STANTemporalTransformer(folder)
    for f in glob.glob(output):
      print('user:', f)
      this_ip = f.split("_")[-1][:-4]
      df = pd.read_csv(f)
      tft.push_back(df, agg=agg, transformer=ntt)
      count += 1
    print(count)

def prepare_standata(agg=5, train_folder='stan_data/day1_data', train_output='to_train.csv', 
                      test_folder='stan_data/day2_data', test_output='to_test.csv'):
  if len(train_folder):
    print('making train for:')
    _prepare('stan_data/'+train_output, train_folder+'/*.csv')
  if len(test_folder):
    print('making test for:')
    _prepare('stan_data/'+test_output, test_folder+'/*.csv')

if __name__ == "__main__":
  download_ugr16()
  user_analysis()
  user_selection()
  prepare_standata()