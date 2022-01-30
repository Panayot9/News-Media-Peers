#!/usr/bin/python

#
# Read the input data file, compute the Jaccard Similiarity.
# Then select the domain to generate the graph input file for BP 
#
# Two important threshold: 
#	common IPs (or shared IPs)
#	Jaccard Similarity
#

from sets import Set
import time
import sys
import re


def get_current_time():
        return time.strftime('%Y-%m-%d %X', time.localtime(time.time()))

#####
#
# A simple version func to check if it's an ip format string
# re.match('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', dt['data'])
#
def is_ip_addr(ip):
	ip = ip.strip()
	strarr = re.split(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', ip)
	#print ip, strarr, strarr[0]
	#if len(strarr) == 1 and strarr[0] == ip:
	if len(strarr) >= 2 and strarr[1] == ip:
		return True
	else:
		return False



if len(sys.argv) < 3:
	print 'Usage: python {0} <input_file> <ground_truth_file>'.format(sys.argv[0])
	exit()

infile = sys.argv[1]
ground_truth_file = sys.argv[2]

#two states: s1=malicious, s2=benign
node_belief_bad_dom = [0.75, 0.25]
node_belief_good_dom = [0.25, 0.75]
node_belief_unknown_dom = [0.5, 0.5]

node_belief_dict = {'G':node_belief_good_dom,
		    'B':node_belief_bad_dom,
		    'U':node_belief_unknown_dom}


#outfile = sys.argv[2]
#outfile = 'domain_list.txt'
output_graph_nodes_file = 'bp_graph_initial_belief.txt'
output_graph_edges_file = 'bp_graph_edges.txt'

output_graph_number_dom = 'bp_graph_node_index.txt'

domain_nodes_list = [] #for all the nodes in the graph
grdtruth_dict = {} #for ground truth list

domain_set = Set() #for all the nodes in the graph
#dom_ip_dict = {} #for edge [dom]=>[ip]
node_index_dict = {} #for dom/ip nodes [dom|ip]=>[index]

s1 = Set() #tmp for nodes(domain)
nodes_cnt = 0
edges_cnt = 0
dom_ip_list = [] #for edges: [dom ip]

malicious_domains_in_graph_cnt = 0
benign_domains_in_graph_cnt = 0
unknown_domains_in_graph_cnt = 0
unknown_ips_in_graph_cnt = 0

exception_line_log_file = 'exception_line_log'
exception_line_cnt = 0

dom_cnt = 0
#only for test
#print('common_ips_threshold: {0}\njaccard_threshold: {1}'.format(common_ips_threshold, jaccard_threshold))
print node_belief_dict, node_belief_dict['G'][0], node_belief_dict['G'][1]
print '\n'


with open(ground_truth_file, 'r') as indata:
	for line in indata:
		arr = line.strip().split(' ')
		if arr:
			dom = arr[0].strip()
			val = arr[1].strip() #val='G', 'B', or 'U'
			if dom != '' and val != '':
				grdtruth_dict[dom] = val
			else:
				print line, arr
		else:
			print line

output_nodes = open(output_graph_nodes_file, 'w')
output_edges = open(output_graph_edges_file, 'w')
exception_log = open(exception_line_log_file, 'w')
output_graph = open(output_graph_number_dom, 'w')

start = time.time()
with open(infile, 'r') as indata:
	for line in indata:
		arr = line.strip().split(' ')
		if arr:
			try:
				dom = arr[0].strip()
				if dom.endswith('.'):
					dom = dom[:-1]
	
				ip = arr[1].strip()
				if ip.endswith('.'):
					ip = ip[:-1]

			except:
				exception_line_cnt += 1
				print line, arr
				exception_log.write('{0}'.format(line))
				continue

			#check the common ips
			#if comm_ips <= common_ips_threshold:
			#	continue

			#check the Jaccard Similarity
			#jaccard = float(comm_ips) / float((dom1_ips + dom2_ips - comm_ips))
			#print 'jaccard: ', jaccard

			#if jaccard <= jaccard_threshold:
			#	continue

			domain_set.add(dom)
			domain_set.add(ip)

			#dom_ip_dict[dom] = ip
			dom_ip_list.append('{0} {1}'.format(dom, ip))
		else:
			print line, arr

nodes_cnt = len(domain_set)


end1 = time.time()
elapsed = end1 - start
print 'Counting all nodes using set finished: ', elapsed
start2 = time.time()

index = 0
for item in domain_set:
	#domain_nodes_list.append(item)
	node_index_dict[item] = index

	try:
		grd_val = grdtruth_dict[item]
	except:
		grd_val = 'U'

	#nodes initial belief
	output_nodes.write('{0} s1 {1}\n'.format(index, node_belief_dict[grd_val][0]))
	output_nodes.write('{0} s2 {1}\n'.format(index, node_belief_dict[grd_val][1]))
	#output_nodes.flush()
	if is_ip_addr(item):
		unknown_ips_in_graph_cnt += 1
	else:
		dom_cnt += 1
		if grd_val == 'B':
			malicious_domains_in_graph_cnt += 1
		elif grd_val == 'G':
			benign_domains_in_graph_cnt += 1
		else:
			unknown_domains_in_graph_cnt += 1

	#save [index, dom/ip, G/B/U] into a file
	output_graph.write('{0} {1} {2}\n'.format(index, item, grd_val))		
	index += 1

end2 = time.time()
elapsed = end2 - start2
print 'Generating graph initial value and nodes index files finished: ', elapsed
start3 = time.time()

#for key,val in dom_ip_dict.iteritems():
#	#generate the graph input files: edges
#	output_edges.write('{0} {1}\n'.format(node_index_dict[key], node_index_dict[val]))
#	edges_cnt += 1
for item in dom_ip_list:
	arr = item.split(' ')
	#generate the graph input files: edges
	output_edges.write('{0} {1}\n'.format(node_index_dict[arr[0]], node_index_dict[arr[1]]))
	edges_cnt += 1

output_nodes.close()
output_edges.close()
exception_log.close()
output_graph.close()

end = time.time()
elapsed = end - start3
print 'Generating graph edges file finished: ', elapsed

elapsed = end - start
print 'Total time: ', elapsed


all_unknown_nodes_in_graph_cnt = unknown_domains_in_graph_cnt + unknown_ips_in_graph_cnt
print 'Indexed nodes file: \n\t{0}, all nodes:{1}'.format(output_graph_number_dom, nodes_cnt)
print 'Graph information: \n\t#nodes: {0}, (dom: {5})(Good: {1}, Bad: {2}, Unknown: {3} (dom: {6}))\n\t#edges: {4}'.format(nodes_cnt, benign_domains_in_graph_cnt, malicious_domains_in_graph_cnt, all_unknown_nodes_in_graph_cnt, edges_cnt, dom_cnt, unknown_domains_in_graph_cnt)
print 'Graph input file: \n\t{0}\n\t{1}'.format(output_graph_nodes_file, output_graph_edges_file)
print '\nException lines in the inputfile: {0}'.format(exception_line_cnt)

"""
output = open(outfile, 'w')
for elem in s1:
	output.write('{0}\n'.format(elem))
output.close()

print 'Output file: {0}\nAll domains: {1}'.format(outfile, len(s1))
"""

