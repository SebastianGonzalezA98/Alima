from bs4 import BeautifulSoup
import requests
from datetime import timedelta, date, datetime
import json

def daterange(date1, date2):
    """ Function to create an iterable range of dates.

    Args:
        date1 (datetime.date): starting date
        date2 (datetime.date): ending date

    Yields:
        generator: date object to iterate one value at a time
    """

    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

def create_urls(start_date,end_date):
    """ Function to create an ordered list of urls from
        http://www.economia-sniim.gob.mx/Consolidados.asp?,
        initializing at a starting date, finishing at ending date.

    Args:
        start_date (datetime.date): starting date
        end_date (datetime.date): ending date

    Returns:
        list of string: where every element is an url
    """

    public_holidays = ['2021-01-01','2021-02-01','2021-03-15','2021-04-01','2021-04-02','2021-05-05','2021-09-16-','2021-10-12','2021-11-02',
                   '2021-11-15','2022-01-01','2022-02-07','2022-03-21','2022-04-14','2022-04-15']

    labor_days = []

    # Exclude public holidays and weekends
    weekdays = [5,6]
    for dt in daterange(start_date, end_date):
        if (dt.weekday() not in weekdays) and (str(dt) not in public_holidays):
            labor_days.append(dt.strftime("%Y-%m-%d"))

    urls_list = []

    # Create urls
    for date in labor_days:
        if len(date.split('-')[2]) > 1:
            day = date.split('-')[2]
        else:
            day = date.split('-')[2].replace('0','')
        month = date.split('-')[1]
        year = date.split('-')[0]
        urls_list.append(f'http://www.economia-sniim.gob.mx/Consolidados.asp?prod=&punto=100&edo=&dqdia={day}&dqmes={month}&dqanio={year}&aqdia={day}&aqmes={month}&aqanio={year}')
    
    return urls_list

def max_len_table(x):
    """ Function to iterate through all of the tables of an html code,
        find the index of the table with most table data cell elements,
        and return these.

    Args:
        x (bs4.element.ResultSet): beautiful soup object that contains all tables from the html code

    Returns:
        bs4.element.ResultSet: beautiful soup object that contains the most data cell elements (td tags)
    """
    max=0
    for i in range(len(x)):
        current_len = len(x[i])
        if current_len > max:
            max = current_len
            max_index = i
    
    return x[max_index].find_all('td')

def start_stop(x):
    """ Function to find the starting and finishing indexes of the
        table data cells to iterate. With these, the data of fruits
        and vegetables needed can be extracted.

    Args:
        x (_type_): beautiful soup object that contains the most data cell elements (td tags)

    Returns:
        ints: starting index and finishing index
    """
    for i in range(len(x)):
        if START_KEYWORD in ''.join(x[i].findAll(text = True)):
            ini = i+1
        else:
            pass
        if STOP_KEYWORD_1 in ''.join(x[i].findAll(text = True)) or STOP_KEYWORD_2 in ''.join(x[i].findAll(text = True)):
            fin = i
            break
    return ini,fin

def clean_string(text):
    """ Function to clean messy strings (product name strings inside tags).

    Args:
        text (str): dirty string (such as '\n\n\xa0 Acelga \xa0')

    Returns:
        _type_: clean string
    """
    return text.replace('\n','').replace('\xa0','')

def main():
    # Keywords to return start and finish indexes
    global START_KEYWORD,STOP_KEYWORD_1,STOP_KEYWORD_2

    START_KEYWORD = 'DistribOrig'
    STOP_KEYWORD_1 = 'Granos y Semillas'
    STOP_KEYWORD_2 = 'Flores'

    # Dictionary keys
    KEYS_LIST = ['product','min_price','max_price','avg_price','origin','distr']

    start_dt = date(2021,1,1)
    end_dt = date(2022,7,31)
    urls_list = create_urls(start_dt,end_dt)

    # Empty list to store dictionaries
    dict_list = []

    # Iterate through all of the urls to extract their information
    for item in urls_list:

        page = requests.get(item)
        if page.status_code != 200:                                             # Makes sure the request went through
            page.raise_for_status()
        soup = BeautifulSoup(page.text,'html.parser')
        try:
            soup.find('p').getText()                                            # Makes sure theres information on the page
        except:
            pass
        else:
            print('Current url does not contain information. Url: ', item)
            continue
        tables = soup.find_all('table')
        mt = max_len_table(tables)
        start,stop = start_stop(mt)

        # Populate dummy_list with clean strings
        dummy_list = []
        for i in range(start,stop):
            dummy_list.append(clean_string(''.join(mt[i].getText())))

        # Populate dictionary by iterating dummy_list and pairing key values
        dummy_dict={}
        for i in range(0,len(dummy_list),6):
            dummy_dict = {}
            for j in range(6):
                dummy_dict[KEYS_LIST[j]] = dummy_list[i+j]
            dict_list.append(dummy_dict)
    
    with open('dict_list.json','w') as fp:                                      # Generate json file with extracted data
        json.dump(dict_list,fp,indent=2)

if __name__ == '__main__':
    main()