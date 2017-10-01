# Import data from Excel into MySQL using Python
# Reference: http://transcriptvids.com/v/YLXFEQLCogM.html

import xlrd # only support the classic XLS file not the XML-format XLSX file. 
import MySQLdb # in order to connect database # install: pip install mysqlclient



# Open the workbook and define the worksheet
book = xlrd.open_workbook("data1.xlsx") # the name of excel file
sheet = book.sheet_by_name("data") # the name of sheet in the excel file
#sheet = book.sheet_by_index(0)

# Establish a MySQL connection
database = MySQLdb.connect (host="localhost", user="root", passwd="1234", db="inscitecrawler") # You can also set the path to configuration file using read_default_file, so you don’t have to expose your username or password in the code. 

# Get the cursor, which is used to traverse the database, line by line
cursor = database.cursor() # Basically it's used to read the database, line by line, so that you can perform a query.

# Create the INSERT INTO sql query
query = """INSERT INTO tb_customrulelist (dirName, area, keyword, Tag, count, exactFlag, dateFormat) VALUES (%s, %s, %s, %s, %s, %s, %s)"""

# Create a For loop to iterate through each row in the XLS file, starting at row 2 to skip the headers
for r in range(1, sheet.nrows): # except for index 0 because it is a category name list
    d = sheet.cell(r,1).value
    
    if d=='ABCAU' or d=='ABCNews' or d=='Aljazeera' or d=='BBC' or d=='Boston' or d=='CNN' or d=='CSmonitor' or d=='EuroNews' or d=='Reuters' or d=='Telegraph' or d=='TheGlobeandmail' or d=='USAToday' or d=='WSJ':  
    
        #if d=='ABCNews' or d=='BBC' or d=='Telegraph' or d=='Aljazeera': continue # dirName = 'cross_val_1set'
        #if d=='ABCAU' or d=='Boston' or d=='Reuters' or d=='TheGlobeandmail': continue # dirName = 'cross_val_2set'
        if d=='CNN' or d=='CSmonitor' or d=='USAToday' or d=='EuroNews' or d=='WSJ': continue    
            
            
        dirName = 'cross_val_3set'
        area = sheet.cell(r,2).value
        keyword = sheet.cell(r,3).value
        Tag = sheet.cell(r,4).value
        count = sheet.cell(r,5).value
        exactFlag = sheet.cell(r,6).value
        dateFormat = sheet.cell(r,7).value

        # Assign values from each row
        values = (dirName, area, keyword, Tag, count, exactFlag, dateFormat)
        
        # Execute sql Query
        cursor.execute(query, values) # to execute the SQL query.  
        


# Close the cursor
cursor.close()
# Commit the transaction
database.commit()
# Close the database connection
database.close()

# Print results
print ""
print "All Done! Bye, for now."
print ""

