from scrapy import Spider, Request 
from tokyo_gig.items import TokyoGigItem
import re

class TokyoGigPastSpider(Spider):
    name = 'tokyo_gig_past_spider'
    allowed_urls = ['http://www.tokyogigguide.com/en/gigs/']
    start_urls = ['http://www.tokyogigguide.com/en/gigs/eventslist?task=archive&start=1'] 
    
    def start_requests(self): 
        num_pages = 419 
        pages = [1] + [i * 50 for i in range(1, num_pages+1)]
        
        page_urls = ['http://www.tokyogigguide.com/en/gigs/eventslist?task=archive&start={}'.format(x) for x in pages] 
        
        for url in page_urls[100:319]: #scraped in batches, due to high volume of pages; change the range to send request to new range of pages  
            yield Request(url = url, callback = self.parse) 
    
    def parse(self, response):
		# Find the total number of pages in the result so that we can decide how many urls to scrape next
        rows = response.xpath('//table[@class = "eventtable"]//tbody/tr') 
        
        for row in rows: 
            title = row.xpath('.//td[3][@headers = "jem_title"]/a/span/text()').extract_first()
            print(title)
            
            venue = row.xpath('.//td[4][@headers = "jem_location"]/a/text()').extract_first()
            print(venue)
            
            category = row.xpath('.//td[5][@headers = "jem_category"]//text()').extract()
            category = [word for word in category if word not in ('\n', ', ', ' ')]
            print(category)
            
            next_page_url = 'http://www.tokyogigguide.com/' + row.xpath('.//td[3][@headers = "jem_title"]/a/@href').extract_first()  
            
            yield Request(next_page_url, meta = {'title': title, 'venue': venue, 'category': category},  callback = self.parse_details_page) 
            
    def parse_details_page(self, response): 
        
        title = response.meta['title']
        venue = response.meta['venue']
        category = response.meta['category'] 
        
        try: 
            date = response.xpath('.//dl[@class = "event_info floattext"]/dd[@class = "when"]/meta[@itemprop = "startDate"]/@content').extract_first()  
            
        except: 
            date = ''
            
        try: 
            
            advance_price = response.xpath('.//dl[@class = "event_info floattext"]/dd[@class = "custom2"]/text()').extract_first() 
                
        except: 
            
            advance_price = '' 
            
        try: 
            door_price = response.xpath('.//dl[@class = "event_info floattext"]/dd[@class = "custom3"]/text()').extract_first() 
            
        except: 
            door_price = '' 
                
        try: 
            address = response.xpath('.//div[@itemprop = "location"]/dl[@class = "location floattext"]/dd[@class = "venue_street"]/text()').extract_first().strip() 
            
            
        except: 
            address = ''
                
        try: 
            area = response.xpath('.//div[@itemprop = "location"]/dl[@class = "location floattext"]/dd[@class = "venue_city"]/text()').extract_first().strip() 
            
        except: 
            area = ''
            
        try: 
            closest_station = response.xpath('.//div[@itemprop = "location"]/dl[@class = "location floattext"]/dd[@class = "custom3"]/text()').extract_first().strip() 
            
        except: 
            closest_station = ''
            
        try: 
            venue_url = response.xpath('.//div[@itemprop = "location"]/dl[@class = "location"]/dd[@class = "venue"]/a/@href').extract()[1] 
            
        except: 
            venue_url = ''
            
        try: 
            access_url =  response.xpath('.//div[@itemprop = "location"]/dl[@class = "location floattext"]/dd[@class = "custom2"]/a/@href').extract_first() 
        except: 
            access_url = '' 
               
        item = TokyoGigItem() 
        item['title'] = title 
        item['date'] = date 
        item['venue'] = venue
        item['venue_url'] = venue_url 
        item['category'] = category
        item['address'] = address
        item['area'] = area
        item['closest_station'] = closest_station 
        item['access_url'] = access_url 
        item['advance_price'] = advance_price 
        item['door_price'] = door_price 
        
        yield item 
             
                
                
		
               
		
        
       
          
            
            
            
            
            
            
            
                       

            
        
            
            