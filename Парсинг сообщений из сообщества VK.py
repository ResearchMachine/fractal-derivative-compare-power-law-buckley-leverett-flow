#Выгрузка сообщений из сообщества VK по ключевым словам в реальном времени

import vk
import time

#--------------------------------------------место для настройки
LinksGroups=['vk.com/pomogashki_mai','vk.com/usatu'] #ссылки на обрабатываемые сообщества
KeyWords=['работа','аналитик','ргр'] #ключевые слова, обязательный чтобы был нижний регистр
StopWords=['21','отдых'] #стоп слова, обязательный чтобы был нижний регистр
CountOfAnalisPosts=5 #число запрашиваемых за раз постов из каждого сообщества для сортировки

#--------------------------------------------вспомогательная часть
MaxRequestsInDay=5000 #сколько можно запросов в день
SecInDay=86400
StopTimeForRequests=len(LinksGroups)*SecInDay/MaxRequestsInDay
print(StopTimeForRequests)

session = vk.Session(access_token='19c33e8f19c33e8f19c33e8f2d19a50672119c319c33e8f4267e6c503786152e3d709e0')
ver='5.0' 
vk_api = vk.API(session)


def getUserId(link): #функция возвращающая id по ссылке 
	id = link
	if 'vk.com/' in link: # проверяем является ли аргумент ссылкой
		id = link.split('/')[-1]  # если да, то получаем его последнюю часть
	if not id.replace('id', '').isdigit(): # если в нем после отсечения 'id' сами цифры - это и есть id 
		id = vk_api.utils.resolveScreenName(v=ver,screen_name=id)['object_id'] # если нет, получаем id с помощью метода API
	else:
		id = id.replace('id', '')
	return int(id)


#b=vk_api.utils.getServerTime(v=ver) текущее время 
#print(b)
#print(time.ctime(b))


def Main(StartTimePosts): 
	global LinksGroups, KeyWords, StopWords, CountOfAnalisPosts
	FinishTimePosts=[] #массив для рассчета последнего обработонного поста для каждой группы
	FinishTimePosts.extend([ 0 for i in range(0,len(LinksGroups))])

	j=0
	for OneLinkGroup in LinksGroups: #цикл по группам
		IdGroup=getUserId(OneLinkGroup)
		posts = vk_api.wall.get(v = ver, owner_id =-IdGroup, count = CountOfAnalisPosts)['items']
		
		FinishTimePosts[j]=posts[0].get('date')
		if posts[0].get('is_pinned')==1: #проверка закреплена ли запись
			FinishTimePosts[j]=posts[1].get('date') #если да, то для вычисления времени последнего поста используется следующий
		print(time.ctime(FinishTimePosts[j]))
		
		if FinishTimePosts[j]>StartTimePosts[j]:
			for i in range(0,CountOfAnalisPosts):	#цикл по по постам группы
				TextPost=posts[i].get('text').lower() #текст поста в нижний регистр
				for OneKeyWord in KeyWords: #проверка вхождения ключевых слов
					IterForStop=-1 #иттератор для стоп слов
					for OneStopWord in StopWords: #проверка НЕ вхождения стоп слов
						IterForStop+=TextPost.find(OneStopWord)
						
					if TextPost.find(OneKeyWord)>0 and IterForStop<0:
						a=OneLinkGroup+'?w=wall-'+str(IdGroup)+'_'+str(posts[i].get('id')) #вернем ссылку на запись
						print(a)
						print(TextPost)
						print('----------------------')
		j+=1
	return FinishTimePosts

StartTimePosts=[] 
StartTimePosts.extend([ 0 for i in range(1,len(LinksGroups)+1)])		
for i in range(0,3):	
	a=Main(StartTimePosts)	
	StartTimePosts=a
	print(time.ctime(a[0]),time.ctime(a[1]))
	time.sleep(1)
	#time.sleep(StopTimeForRequests)
	
#print(time.ctime(a[0]),time.ctime(a[1]))	
#time.ctime(FinishTimePosts)
#

