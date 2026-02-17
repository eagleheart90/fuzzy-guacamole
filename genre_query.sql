SELECT (year/10)* 10 AS decade,
COUNT(*)as horror_movie_count
FROM movies
WHERE genres LIKE '%Horror%'
GROUP BY decade
ORDER BY decade ASC;