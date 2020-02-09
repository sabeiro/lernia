var data, options, heatmap;
function loadSet(jData) {
    var json = null;
    $.ajax({
	async: false,
	global: false,
	url: jData,
	dataType: "json",
	success: function (data) {
            json = data;
	},
	failure: function() {alert("problem with the file" + jData); }
    });
    return json;
};
//data = loadSet('data/networkPartyLinks.json');
//data = loadSet('data/networkTaxonomyLinks.json');
//data = loadSet('data/func_link.json');
//data = loadSet('data/networkSkillLink.json');
//data = loadSet('data/network.json');
//var links = data.links;
//data = loadSet('data/networkPartyNodes.json');
var data = loadSet('data/func_network.json');
//data = loadSet('data/networkSkillNode.json');
//data = loadSet('data/networkNode.json');
var nodes = data.nodes;
var width = 1400
height = 800;
var color = d3.scale.category20();
var force = d3.layout.force()
    .nodes(d3.values(data.nodes))
    .links(data.links)
    .size([width, height])
    .linkDistance(200)
//    .charge(-15000)
    .charge(-300)
//    .gravity(1)
    .on("tick", tick)
    .start();

force.linkDistance(function(links){
    return links.distance;
});
force.charge(function(nodes){
    return nodes.size;
})

// var simulation = d3.forceSimulation(nodes)
//     .force("charge", d3.forceManyBody())
//     .force("link", d3.forceLink(links).distance(20).strength(1))
//     .force("x", d3.forceX())
//     .force("y", d3.forceY())
//     .on("tick", ticked);

var svg = d3.select("body").append("svg")
    .attr("width", width)
    .attr("height", height);

var link = svg.selectAll(".link")
    .data(force.links())
    .enter().append("line")
    .style("opacity",.1)
    .attr("class","link")
    .style("stroke-width", function(d) { return Math.sqrt(d.value); });

var node = svg.selectAll(".node")
    .data(force.nodes())
    .enter().append("g")
    .attr("class", "node")
    .style("fill", function(d) { return color(d.group); })
    .style("opacity", 0.9)
    .on("mouseover", mouseover)
    .on("mouseout", mouseout)
    .call(force.drag);

node.append("circle")
    .style("opacity", .5)
    .attr("r",function(d) { return 6*Math.sqrt(d.size) } )
//    .attr("r", 6)

node.append("svg:text")
    .attr("x", 13)
    .attr("class", "nodetext")
    .attr("dx", 12)
    .attr("dy", ".35em")
    .style("stroke-width", ".5px")
    .style("font", function(d) {return 6*Math.sqrt(d.size) + "px serif" })
    .style("opacity", 1)
    .text(function(d) {return d.name});


function distance(){
    return 30;
}

function tick() {
    link
	.attr("x1", function(d) { return d.source.x; })
	.attr("y1", function(d) { return d.source.y; })
	.attr("x2", function(d) { return d.target.x; })
	.attr("y2", function(d) { return d.target.y; });

    node.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
}

function mouseover() {
    // d3.select(this).select("circle").transition()
    // 	.duration(750)
    // 	.attr("r", 16);
    
    // d3.select(this).select("text").transition()
    // 	.duration(750)
    // 	.attr("x", 13)
    // 	.style("stroke-width", ".5px")
    // 	.style("font", "8.5px serif")
    // 	.style("opacity", 1);
}

function mouseout() {
    // d3.select(this).select("circle").transition()
    // 	.duration(750)
    // 	.attr("r", 8);
}
function onclick() {
    d3.select(this).select("circle").transition()
	.duration(750)
	// .attr("r", 8);
    d3.select(this).select("text").transition()
	.duration(750)
	.attr("x", 13)
	.style("stroke-width", ".5px")
	// .style("font", "17.5px serif")
	.style("opacity", 1);
}
