# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class TokyoGigItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    title = scrapy.Field()
    date = scrapy.Field() 
    venue = scrapy.Field()
    venue_url = scrapy.Field() 
    category = scrapy.Field() 
    address = scrapy.Field() 
    area = scrapy.Field() 
    closest_station = scrapy.Field() 
    access_url = scrapy.Field() 
    advance_price= scrapy.Field() 
    door_price = scrapy.Field() 
    
    
    
    
    
